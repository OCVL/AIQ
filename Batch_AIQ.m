clear;
close all;
basePath = uigetdir(pwd);

fNames = read_folder_contents_rec(basePath,'tif');

for f=1:length(fNames)
    
    
    im = imread(fNames{f});
    
    IQ(f) = AIQ_DFT(im);
    
    
end

outtable = table(fNames,IQ');


writetable(outtable, fullfile(basePath, 'IQ_Metrics.csv'));