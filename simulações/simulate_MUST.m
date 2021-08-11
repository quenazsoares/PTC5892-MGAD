%% Introdução
% A toolbox MUST realiza simulações de ultrasom modo B.
% É possível personalizar diversos parâmetros, porém algums serão deixados
% no padrão correspondente a equipamentos comerciais.

%% Padrões de imagens
% checkboard: Quadriculado com 256x256 pixels. Pixels escuros são cinza 10%
% e pixels claros são cinza 50%.
%
% cistos: Duas colunas de 5 circulos com tamanhos diferentes. Circulos
% escuros são preto (valor 0) e circulos claros são branco (valor 255).
% Fundo é cinza 50% (valor 128).
%
% forms: 4 formas com níveis de cinza distintos. elipse (209), retângulo
% (53), meia coroa circular com cantos arredondados (14) e trevo (248).
% Fundo é 131. A imagem é de 32 bits.

%%%%%%%%%%%   IMPORTANTE: Instaltar MUST via toolbox manager!   %%%%%%%%%%%%%%%%%%

%%
clear
clc
multiWaitbar( 'CloseAll' );

%% Configurações
% Adiciona a toolbox MUST ao PATH
addpath("%appdata%\MathWorks\MATLAB Add-Ons\Toolboxes\MUST")

input_imgs = ["goldstd\forms.tif", "goldstd\checkboard2.tif", "goldstd\cistos.tif"];
%input_imgs = ["goldstd\checkboard3.png"];
%input_imgs = ["goldstd\forms.tif"];
%parameters = ["P4-2v", "C5-2v", "L11-5v"];
%parameters = ["C5-2v", "L11-5v", "P4-2v"];
parameters = ["C5-2v"];
n_sims = 15;
n_steps = 5;

scatdens = 1.5; % scatterer density per lambda^2 (you may modify it, default: 1.5)
%scatdens = 3;
output_dir_base = "speckled_NEW_2"; %"speckled";
show_images = true;

compound_type = "polar"; % Tipo de composição ("polar" ou "linear")

%cmap = colormap('hot');
cmap = colormap('hsv');
cmap(193:end,:) = [];
%% Verificações dos parâmetros de entrada
for file=input_imgs
    assert(exist(string(pwd) + filesep + file, 'file'), "Verifique se o arquivo " + file + " existe!")
end

%% Algoritmo
count_parameters = 0;
multiWaitbar( 'Parâmetros do ultrassom', 0);
for param_name = parameters
    param = getparam(param_name);
    output_dir = fullfile(pwd, output_dir_base, param_name);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % Center wavelength.
    param.c = 1540; % speed of sound (m/s)
    lambda = param.c/param.fc;
    
    count_files = 0;
    for sim = 1:n_sims
        multiWaitbar( 'Simulacoes', 0);
        for file = input_imgs
            multiWaitbar( 'Modelos de Phantom', 0);
            [~,filename,~] = fileparts(file);

            % Load the image
            Ig = imread(file);
            IgUD = flipud(Ig);
            %Ig = (imread(file))/255;
            %Ig = imnoise(Ig,'speckle',0.1^2)*255;
            z_init = 0.10; % Distância até o início do phantom
        
        
            
            % Simulate scatterers in a 5-cm ${\times}$ 5-cm region.
            % Image grid for a 5-cm deep image
            [nl,nc,~] = size(Ig);
            L = 5e-2;
            [xi,zi] = meshgrid(linspace(0,L,nc)*nc/nl,linspace(0,L,nl));
            xi = xi-L/2*nc/nl; % recenter xi
            zi=zi+z_init;
            
            % Obtain randomly distributed scatterers by interpolation.
            Ns = round(scatdens*L^2*nc/nl/lambda^2); % number of scatterers
            
            xs = rand(1,Ns)*L-L/2; % scatterer locations
            zs = rand(1,Ns)*L+z_init ;

            F = scatteredInterpolant(xi(:),zi(:),double(Ig(:))/1024);
            
            %g = 0.4; % this parameter adjusts the RC values
            signal = randi(2,size(xs))*2-3;
            % G = [0.2 0.25 0.3];
            G = 0.4;
            count_G = 0;
            for g = G
                multiWaitbar( 'G', 'Value', 0);
                RC = F(xs,zs).^(1/g); % reflection coefficients
                RC = RC .* signal;
                noise = randn(size(RC))*(0.25^2);
                RC = RC+noise.*RC;
                if show_images
                    figure(1)
                    scatter(xs*1e2,zs*1e2,3,RC,'filled')
                    axis equal ij tight
                    colormap(cmap)
                    set(gca,'XColor','none','box','off')
                    ylabel('[cm]')
                    title('Scatterers and their reflection coefficients')
                end
                
                [xk,zk] = meshgrid(linspace(0,L,nc)*nc/nl,linspace(0,L,nl));
                xk=xk(:);
                zk=zk(:);
                xk = xk-L/2*nc/nl;
                
                if strcmpi(compound_type, 'polar')
                    % Simulate n_steps sets of RF signals obtained with n_steps circular waves tilted at different angles, beamform these signals onto a 128 ${\times}$ 128 polar grid, and obtain a compound I/Q dataset.
                    tilt = linspace(-pi/6,pi/6,n_steps); % tilt angles
                else
                    % Simulate n_steps of RF signals obtained with n_steos circular waves shifted at diferent x positions.
                    x_shift = linspace(0,L,n_steps)-L/2;
                end
                
                IQc = zeros(nl,nc,'like',1i); % will contain the compound I/Q
                
                opt.WaitBar = true; % no progress bar for SIMUS
                param.fs = param.fc*4; % RF sampling frequency
                xI = xi;
                zI= zi;
                
                multiWaitbar( 'RF signals', 0);
                for k = 1:n_steps
                    
                    if strcmpi(compound_type, 'polar')
                        dels = txdelay(param,tilt(k)); % transmit delays with polar tilts
                    else
                        dels = txdelay(x_shift(k),0,param); % transmit delays with shifts along x axis
                    end
                    
                    RF = simus(xs,zs,RC,dels,param,opt); % RF simulation
                    IQ = rf2iq(RF,param); % I/Q demodulation
                    IQb = das(IQ,xI,zI,dels,param); % DAS beamforming
                    IQc = IQc+IQb; % compounding
                    multiWaitbar( 'RF signals', 'Value', k/n_steps);
                end
                
                
                lcI = bmode(IQc,50); % log-compressed image
                
                out_path = fullfile(output_dir, sprintf("%s_%02d_g%.2f", filename, sim, g));
                % Salva imagem compound I/Q
                save(out_path + ".mat", "IQc")
                % Salva a imagem em B-mode
                imwrite(lcI, out_path + ".png", 'png');
                
                if show_images
                    figure(2)
                    pcolor(xI*1e2,zI*1e2,lcI)
                    shading interp, axis equal ij tight
                    c = colorbar;
                    c.YTick = [0 255];
                    c.YTickLabel = {'-50 dB','0 dB'};
                    colormap gray
                    axis equal tight
                    ylabel('[cm]')
                    set(gca,'XColor','none','box','off')
                    title('Ultrasound compound image')
                end
                fprintf("phanton: %s  \t  sim: %d  \t  g: %.2f\n", filename, sim, g)
                count_G = count_G + 1;
                multiWaitbar( 'G', 'Value', count_G/length(G));
            end
            
            count_files = count_files + 1;
            multiWaitbar( 'Modelos de Phantom', 'Value', count_files/length(input_imgs));    
        end
        multiWaitbar( 'Simulacoes', 'Value', sim/n_sims);
    end
    
    count_parameters = count_parameters + 1;
    multiWaitbar( 'Parâmetros do ultrassom', 'Value', count_parameters/length(parameters) );
end