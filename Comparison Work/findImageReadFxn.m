function [fhandle] = findImageReadFxn(shot)
%% fhandle = findImageReadFxn(shot)
% 
% Input:
%     shot - the test number whose images are to be processed
% 
% Output:
%     fhandle - the handle to the function which will properly read the
%         images from the specified shot. A sample function call is shown
%         below. If shot does not corrospond to any of the avaliable data
%         fhandle is returned empty.
% 
% [I,BLmethod,rotate] = fhandle(base_dir,shot,img_no)
% or
% [I,BLmethod,rotate,dx] = fhandle(base_dir,shot,img_no)
% 
% base_dir - the directory in which the images folder is found.
% img_no - the specific image to be read
%
% I - the image as a uint## matrix, not converted to double.
% BLmethod - the method to be used to find the location and extent of the
%       boundary layer
% rotate - logical value indicating whether the image needs to be rotated
% dx - distance between centers of each pixel
%
% If additional image sequences are added this function must be modified to
% account for them and a new function may have to be written.

stuartShots = [1296 1297 1307 1308 1432 1433 1438:1441 1443 1447];
T9Shots = [3745:3750 3754:3759 3761 3764:3766 4015:4022 4117:4123 450:453 4169];
larc_shots = [7025];
wpafb_shots = [91, 92]; % [9, 16, 30] deleted bc overlapped with
%Langley
Langley_shots = [2:81];
%afrl shots included in t9 function just because that's what we've been using lately
exampleShot = 1;

fhandle = [];

if regexp(sprintf('%s', shot), 'dmd')
   fhandle = @readImagesDirect; 
   return;
end

if sum(stuartShots==shot) > 0
    fhandle = @readImagesStuart;
end

if sum(exampleShot==shot)>0
    fhandle = @readExampleImg;
end

if sum(T9Shots == shot) > 0
    fhandle = @readImagesT9;
end

if sum(larc_shots == shot) > 0
    fhandle = @readImagesLarc;
end

if sum(Langley_shots == shot) > 0
    fhandle = @readImagesLangley;
end

end