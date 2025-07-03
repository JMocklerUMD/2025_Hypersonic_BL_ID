clc; clear; close all

%% Establish params and read in images
% plots the power spectrum from a single row in a single image

% Image and control parameters
row_range = 1:64;           % Rows to keep
col_range = 1:1280;         % cols to keep
scale_fact = 20;            % scale up the final image for contrast?
flip_image = 1;             % Flip the image to compensate
use_filtfilt = 0;           % Use MATLAB's filter instead?
subtract_mean_frame = 1;    % Pre-subtract mean frame? Maybe not necessary?

% Filtering parameters
cut_low = 0.2; cut_high = 0.010;   % Tune these numbers in
order_low = 8; order_high = 8;      % Don't do much higher than 8 for numerical stability

% Physical parameters
x0 = 594; % s value in mm corresponding to pixel 1
% ind = find(shot_all == shot);
% mm_pix = 1/pix_mm_all(ind);
mm_pix = 1;
dx = mm_pix*1e-3; % distance between pixels in mm (m?)    

% File handling
% data_file = 'C:\Users\stuartl\research\HEG\boundary_layer_viz\shot_data\shot_data_phantom.txt';
% fid = fopen(data_file,'r');
% A = textscan(fid,'%d %f %f %f %f','Headerlines',1);
% fclose(fid);
% shot_all = A{1}; fr_all = A{2}; orient_all = -A{3}; pix_mm_all = A{4}; y0_all = A{5};


%% Load in images and create mean image

folder_path = 'C:\Users\Joseph Mockler\Documents\GitHub\2025_Hypersonic_BL_ID\filtering_test_imgs';
images = dir(fullfile(folder_path, '*.tif'));
image_list = cell(1, length(images));
for i = 1:length(images)
    path = fullfile(folder_path, images(i).name);
    image_list{i} = imread(path);
end

% Form the mean image
mean_img = double(image_list{1});
for i = 2:length(image_list)
    img = double(image_list{i});
    mean_img = mean_img + img;
end
mean_img = mean_img / length(image_list);

% Background-subtract the frames
processed = cell(1, length(images));
for i = 1:length(images)
    img = double(image_list{i});
    % Only subtract mean image if selected ahead of time
    if subtract_mean_frame == 1
        img = img - mean_img;
    end
    processed{i} = img;
end
nimg=length(images);


%% Filtering
% Build a new filter in case we want to use MATLAB built-in
[b,a] = butter(8, [cut_high, cut_low]);

tic
% Filter each image
for ii = 1:nimg
  % Set an empty image and read in the current image
  place_holder = zeros(max(row_range), max(col_range));
  In=processed{ii};

  % Process each row
  for jj = 1:length(row_range)
    if use_filtfilt == 1
        vip = filtfilt(b, a, In(jj,:)');
    else
        vip = filter_notch(In(jj,:)',cut_low,cut_high,order_low,order_high);
    end
    % Populate the new image, row by row 
    place_holder(jj,:) = vip';
  end
  
  % Flip only if using Dr. Laurences filter.
  if flip_image == 1 & use_filtfilt == 0
    %In = imflip16(place_holder(jj,:),'v');
    place_holder = flip(place_holder, 2);
  end
  
  % Save off the frame
  If{ii} = place_holder;
  If{ii} = rescale(If{ii},0,1);
  
end
toc

% Inspect a random frame
r=randi([1,nimg]);
processed{r} = rescale(processed{r}, 0, 1);
figure 
subplot(3,1,1)
imshow(processed{r})
title('original')

subplot(3,1,2)
imshow(If{r})
title('filtered')
diff=If{r}-processed{r};

subplot(3,1,3)
imshow(diff)
title('difference')
sgtitle(['Frame ' num2str(r)])


% Show how the filtering altered a specific line
figure
plot(col_range, processed{r}(45,:)); hold on
plot(col_range, If{r}(45,:));
plot(col_range, diff(45,:))
title('Row 45 (wave packet range) filtered signal')
legend('Original', 'filtered', 'difference')
grid minor;


% x = x0 + (col_range - col_range(1))*mm_pix;
% 
% figure('Position',[100,100,1100,400])
% gap = 0.02;
% height = (1-nimg*gap)/nimg;
% for ii = 1:2
%   yloc = gap/2 + (nimg - ii)*(height + gap);
%   axes('Position',[0.02,yloc,0.96,height])
%   imshow(If{ii})
% end
% %colormap hot