%% Create a video from png frames

clear

% File path to set of images
folder_path = 'C:\Users\Joseph Mockler\Documents\GitHub\2025_Hypersonic_BL_ID\Re33_testplot';

% read in files in folder
image_dir = dir(fullfile(folder_path, '*.png'));

% read in image data 
images = cell(1, length(image_dir));
for i = 1:length(image_dir)
    path = fullfile(folder_path, image_dir(i).name);
    images{i} = imread(path);
end

% Writes to the matlab working directory
writerObj = VideoWriter([folder_path, '\myVideo4.avi']);
writerObj.FrameRate = 6; % in frames / sec

% open the video writer
open(writerObj);

% write the frames to the video
for u=1:length(images)
 % convert the image to a frame
 frame = im2frame(images{u});
 writeVideo(writerObj, frame);
end

% close the writer object
close(writerObj);
