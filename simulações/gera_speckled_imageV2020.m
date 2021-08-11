% gera_speckled_imageV2020.m   gerador de ruido speckle  S Furuie 5/12/2020
% %%%%%%%%%%%%%%%%%
% Precisa do image processing toolbox: 
% imnoise
% impixelinfo
% %%%%%%%%%%%%%%%%
cvs=[0.05, 0.1, 0.3 0.5];            %coef de variacao
cvs = 0.5;
input_imgs = ["goldstd\checkboard3.png", "goldstd\cistos.tif", "goldstd\forms.tif"];
n_sims = 50;
output_dir = "speckled_SIMPLE";

 for file = input_imgs
    [~,filename,~] = fileparts(file);
    % Load the image
    I = imread(file);
    D = uint16(I)+16;
    for n=1:n_sims
        count = 1;
        for cv = cvs
            variancia = cv^2;
            Ispeckled = imnoise(D,'speckle',variancia);
            arquivoOut = sprintf('%s_cv=%04.2f_%02d.tif',filename,cv, n);
            arquivoOut = fullfile(pwd, output_dir, arquivoOut);
            imwrite(Ispeckled,arquivoOut,'tiff');  % salvar como 16 bits p/ evitar saturacao
            fprintf("%s - %04.2f %04.2f: \t", filename, n/n_sims, count/length(cvs))
            fprintf("Arquivo salvo em: %s\n", arquivoOut)
            count = count+1;
        end
    end
 end
fprintf("Fim!\n")

