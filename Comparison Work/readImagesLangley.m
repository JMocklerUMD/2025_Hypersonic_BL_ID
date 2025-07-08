function [I,BLmethod,rotate,varargout] = readImagesLangley(base_dir,shot,img_no)
%% I = readImagesLangley(base_dir,shot,img_no)
% [I,dx] = readImagesLangley(base_dir,shot,img_no)
% 
% Inputs:
%     base_dir - the folder that the contains the images folder where all of
%         the images are stored in seperate directories.
%     shot - the shot number, which corrosponds to the number of the specific
%         directory that the images are in.
%     img_no - the particular image that is to be retrieved
% Outputs:
%     I - the raw image returned by imread
%     BLmethod - the method used for finding the boundary layer
%     rotate - indicator as to whether the image needs to be rotated
%     dx - (OPTIONAL) the distance from the center of one pixel to the next
% 
% This function navigates through the directory sturcture to retireve images
% from the Air Force T9 CoTE cone test done at NASA Langley


%COMMENT THIS OUT FOR NOW; NOT NEEDED FOR IDENTIFICATION ANALYSIS
% data = csvread([base_dir '/T9 full.csv']);
% ind = find(data(:,1) == shot,1);
% dx = 0.001/data(ind,2);
% if nargout > 1
%     varargout = {dx};
% end
BLmethod = 1;
rotate = 1;
minImg = 0;

schl_dir = sprintf('run%d_data\\Img%06d.tif',shot,img_no);

I = imread(schl_dir); % load the image
if length(size(I)) == 3
    I = I(:,:,1);
end
end