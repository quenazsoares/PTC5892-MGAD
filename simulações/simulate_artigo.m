% Gera imagens com ruído Rayleigh
% Segundo Yang2004 "Speckle Reduction and Structure Enhancement by Multichannel Median Boosted Anisotropic Diffusion"

input_imgs = ["goldstd\checkboard3.png", "goldstd\cistos.tif", "goldstd\forms.tif"];
n_sims = 50;
output_dir = "speckled_ARTIGO";

for file = input_imgs
    [~,filename,~] = fileparts(file);
    % Load the image
    I = double(imread(file))+16;
    
    for n=1:n_sims
        Inoise = raylrnd(I)-16;
        Ispeckled = imgaussfilt(Inoise, 2, 'FilterSize',5);
        Ispeckled = uint16(Ispeckled);
        arquivoOut = sprintf('%s_%02d.tif',filename, n);
        arquivoOut = fullfile(pwd, output_dir, arquivoOut);
        imwrite(Ispeckled,arquivoOut,'tiff');  % salvar como 16 bits p/ evitar saturacao
        fprintf("%s - %04.2f: \t", filename, n/n_sims)
        fprintf("Arquivo salvo em: %s\n", arquivoOut)
    end
end
fprintf("Fim!\n")