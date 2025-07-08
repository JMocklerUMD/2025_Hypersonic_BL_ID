% Modified Boundary Layer Identification Script
% Takes in a text file input (filtered images)
% Outputs a text file

clc; clear; close all

%% Establish parameters

% Image and control parameters
flip_image = 1;       % Flip the image for correct flow direction (to the right)
boundary_height = 15; % Height of boundary layer in pixels

% Filtering parameters
cut_low = 1/boundary_height; cut_high = 1/(4*boundary_height);
order_low = 8; order_high = 8; % Don't do much higher than 8 for numerical stability

%% Load in images and create mean image

folder_path = 'C:\Users\cathe\Documents\MATLAB\trainML\T9Run4120';
% read in files in folder
images = dir(fullfile(folder_path, '*.tif'));

% read in image data 
image_list = cell(1, length(images));
for i = 1:length(images)
    path = fullfile(folder_path, images(i).name);
    image_list{i} = imread(path);
end

processed = cell(1, length(images));

for k = 1:100:length(images)
    batch_end = min(k+99, length(images));
    % create mean reference image
    mean_img = double(image_list{k});
    tot = 1;
    for i = (k+1):batch_end
        img = double(image_list{i});
        mean_img = mean_img + img;
        tot = tot+1;
    end
    mean_img = mean_img./ tot;
    
    % subtract mean image from others
    for i = k:batch_end
        img = double(image_list{i});
        img = img - mean_img;
        if flip_image == 1
            img = flip(img,2);
        end
        processed{i} = rescale(img, 0, 1);
    end
end

[rows, cols] = size(mean_img);
row_range = 1:rows;
col_range = 1:cols;

%% Filtering

% make sure filter_notch.m is in your path

% Filter each image
for ii = 1:length(images)
  % Set an empty image and read in the current image
  place_holder = zeros(max(row_range), max(col_range));
  In=processed{ii};

  % Process each row
  for jj = 1:length(row_range)
    vip = filter_notch(In(jj,:)',cut_low,cut_high,order_low,order_high);    
    place_holder(jj,:) = vip'; % Populate the new image, row by row
  end

  % Save off the frame
  If{ii} = place_holder;
  If{ii} = rescale(If{ii},0,1);  
end

%% Classification of filtered images 

save_file = fullfile(folder_path, 'filtered_partial_results.mat');

if isfile(save_file)

    load(save_file, 'results', 'start_frame');
    disp(['Resuming from frame ' num2str(start_frame)])
else
    results = cell(length(images), 8);
    start_frame = 1;
end

i = start_frame;
while i <= 10%length(images)
    
    % display image
    img_original=processed{i};
    img = If{i};
    %img_enhanced = imadjust(img_original); 
    subplot(2,1,1)
    imshow(img_original)
    subplot(2,1,2)
    imshow(img)
    set(gcf, 'WindowState', 'maximized')
    % set(gcf, 'Color', 'k');  
    t = title(['Frame ' num2str(i)]);
    t.Color = 'black';
    
    wp = input('Wave packet? (0 = no, 1 = yes, 2 = relabel previous, 3 = throw away, 4 = exit): ');
    
    % throwaway
    if wp == 3
        results{i,2} = wp;
        i = i+1;
        close;
        continue;
    
    % exit and save
    elseif wp == 4
        disp('Exiting and saving progress...');
        save(save_file, 'results', 'start_frame')
        close;
        break;

    % go back
    elseif wp == 2
       i = i-1;
       close;
       continue;

    % there is a wavepacket! draw a bounding box
    elseif wp == 1
        disp('Draw bounding box around the wave packet.');
        bb = drawrectangle(); 
        bbox = bb.Position; % [x y width height]
        results{i,3} = bbox(1);
        results{i,4} = bbox(2);
        results{i,5} = bbox(3); 
        results{i,6} = bbox(4);

    % there is no wavepacket
    elseif wp == 0
        results{i,3} = 'X';
        results{i,4} = 'X';
        results{i,5} = 'X'; 
        results{i,6} = 'X';

    % fail safe for invalid inputs
    else
        disp("Invalid input. Try again.")
        continue;
    end

    % store results
    results{i,1} = i;
    results{i,2} = wp;
    img = If{i};
    [rows, cols] = size(img);
    results{i,7} = rows;
    results{i,8} = cols;
    close;
    start_frame = i + 1;
    i = i + 1;
end

%% Write trained data to .txt file

output_file = fullfile(folder_path, 'filtered_training_data.txt');
fileID = fopen(output_file, 'w');

% only perform if all images are labeled
if 11%i == length(images) + 1 % remove if statement if training stopped early and run section
    for k = 1:10%length(images)

        % do not record throwaway images
        if results{k,2} == 3
           continue;
        end
        
        % write run
        fprintf(fileID, '%d\t', results{k,1});
        % write WP ID
        fprintf(fileID, '%d\t', results{k,2});
        
        % write bounding box coords
        for j = 3:6
            item = results{k,j};
            if ischar(item) || isstring(item)
                fprintf(fileID, '%s\t', item);
            else
                fprintf(fileID, '%.0f\t', item);
            end
        end
        
        % write image size [height, width]
        fprintf(fileID, '%d\t%d\t', results{k,7}, results{k,8});
        
        %write image data
        img = If{k};
        img_flat = reshape(img', 1, []);
        for m = 1:length(img_flat)
            pixel = img_flat(m);
            fprintf(fileID, '%.6f\t', pixel);
        end
        fprintf(fileID, '\n');
    end
    fclose(fileID);
    
    % delete saved file when all data is written
    if k == length(images) && isfile(save_file)
        delete(save_file);
    end
end

%% Save raw, labeled as .tiff to folder

outfolder = 'Labeled Images'; %create this folder inside the orignal folder of .tiff images before running

for i = 1:10%length(images)

     if results{i,2} == 3
           continue;
     end
     img = image_list{i};
     imgName = sprintf('img_%03d.tiff', i);
     full_tiff_path = fullfile(outfolder, imgName);
     imwrite(img, full_tiff_path)
end
