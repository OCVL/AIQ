function [ snr ] = AIQ_DFT(test_image)

% Copyright (C) 2019 Robert F Cooper
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%
% clear
% close all

if ~exist('test_image','var') || isempty(test_image)
    [filename, pathname] = uigetfile('*.tif', 'Pick an image to segment');

    test_image = double(imread( fullfile(pathname, filename) ));
    
    immax= max(test_image(:));
%     test_image = test_image./immax;
%     test_image = imnoise(test_image, 'gaussian',0, 0.01).*immax;
    
%     figure(9); imagesc(test_image); colormap gray; axis image;
end

% tic;
if size(test_image,3) >1
    test_image = test_image(:,:,1);
end
im_size = size(test_image);

% if ~exist('roi_size','var') 
    roi_size = round(min(im_size)/4);
% end

if ~exist('supersampling','var')
    supersampling = false;
end

if ~exist('row_or_cell','var')
    row_or_cell = 'cell';
end

% if ~exist('roi_step','var')
    roi_step = floor(roi_size/2); % 50% overlap
% end
interped_spac_map=[];

imcomps = bwconncomp( imclose(test_image>0,ones(5)) );
imbox = regionprops(imcomps, 'BoundingBox');


boxsizes = zeros(size(imbox,1),1);
for i=1:size(imbox,1)
    boxsizes(i)= imbox(i).BoundingBox(3)*imbox(i).BoundingBox(4);
end   
[~, maxsizeind]=max(boxsizes);
imbox = floor(imbox(maxsizeind).BoundingBox);

imbox(imbox<=0) = 1;
width_diff = im_size(2)-(imbox(1)+imbox(3));
if width_diff  < 0 
    imbox(3) = imbox(3)+width_diff;
end
height_diff = im_size(1)-(imbox(2)+imbox(4));
if height_diff  < 0 
    imbox(4) = imbox(4)+height_diff;
end


% Our roi size should always be divisible by 2 (for simplicity).
if rem(roi_size,2) ~= 0
    roi_size = roi_size-1;
end
roi = cell(round((size(test_image)-roi_size)/roi_step));

for i=imbox(2):roi_step:imbox(2)+imbox(4)-roi_size
    for j=imbox(1):roi_step:imbox(1)+imbox(3)-roi_size

        numzeros = sum(sum(test_image(i:i+roi_size-1, j:j+roi_size-1)<=10));

%             if numzeros < (roi_size*roi_size)*0.05
            roi{round(i/roi_step)+1,round(j/roi_step)+1} = double(test_image(i:i+roi_size-1, j:j+roi_size-1));
%             else
%                 roi{round(i/roi_step)+1,round(j/roi_step)+1} =[];
%             end
    end
end

numind = size(roi,1)*size(roi,2);
pixel_spac = nan(size(roi));
confidence = nan(size(roi));

        
% Make our hanning window for each ROI.
hann_twodee = hanning(roi_size)*hanning(roi_size)';

fullfourierProfiles = nan([roi_size roi_size length(roi)]);
polarProfiles = nan([360 roi_size length(roi)]);

tic;
for r=1:length(pixel_spac(:))
    if ~isempty(roi{r})        
        

        power_spect = abs(fftshift(fft2( hann_twodee.*roi{r} )./(roi_size^2) )).^2;        
                
        fullfourierProfiles(:,:,r) = power_spect;
    end    
end

welchDFTs = mean(fullfourierProfiles,3,'omitnan');

% figure(1); imagesc(log10(welchDFTs)); axis image;



rhostart=0; % Exclude the DC term from our radial average

rhosampling = .5;
thetasampling = 1;

[polarroi, power_spect_radius] = imcart2pseudopolar(welchDFTs,rhosampling,thetasampling,[],'makima', rhostart);
polarroi = circshift(polarroi,-90/thetasampling,1);
% figure(101); imagesc(log10(abs(polarroi))); axis image;

upper_n_lower = [1:45 136:180]/thetasampling;
right = (46:135)/thetasampling;
upper_n_lower_fourierProfile = mean(polarroi(upper_n_lower,:));
right_fourierProfile = (mean(polarroi(right,:)));
     
% k*(fs/N)
freq_bin_size = rhosampling/size(polarroi,2);
freqBins = (rhostart:size(polarroi,2)-1).*freq_bin_size;



spacing_bins = 0.4./freqBins;
rperange = find(spacing_bins > 12 & spacing_bins <= 20);
conerange = find(spacing_bins > 2.5 & spacing_bins <= 12);
rodrange = find(spacing_bins > 1.25 & spacing_bins <= 2.5);

noiserange = find(spacing_bins > 0 & spacing_bins <= 1.25);
totalrange = find(spacing_bins > 1.25 & spacing_bins <= 20);

% totalpower = freq_bin_size.*sum(right_fourierProfile(totalrange));
noisepower = abs(freq_bin_size.*sum(diff(right_fourierProfile(noiserange))));

snr = 10*log10(abs(freq_bin_size.*sum(diff(right_fourierProfile(totalrange)))) ./ noisepower);

% 
% coneratio = (freq_bin_size.*sum(right_fourierProfile(conerange))) ./ noisepower
% rodratio = (freq_bin_size.*sum(right_fourierProfile(rodrange))) ./ noisepower
% rperatio = (freq_bin_size.*sum(right_fourierProfile(rperange))) ./ noisepower

% figure;
% plot(freqBins,log10(right_fourierProfile));
% plot( diff(log10(right_fourierProfile)));

toc;
end

