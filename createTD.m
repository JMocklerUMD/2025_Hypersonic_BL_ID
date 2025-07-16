% Modified Boundary Layer Identification Script
% Takes in a text file input (filtered images)
% Outputs a text file

clc; clear; close all

%% Establish parameters

% Image and control parameters
flip_image = 0;       % Flip the image for correct flow direction (to the right)
qty_labeled = 865; % desired quantity of labeled data
main_folder_path = "C:\Users\tyler\Desktop\NSSSIP25\run34decimateby1"; % folder containing: folder of raw .tiff images
image_folder_path = "C:\Users\tyler\Desktop\NSSSIP25\run34decimateby1\1000rawImgs"; % folder of raw .tiff images

%% Load in images and create mean image

% read in files in folder
images = dir(fullfile(image_folder_path, '*.tif'));

% read in image data 
image_list = cell(1, length(images));
for i = 1:length(images)
    path = fullfile(image_folder_path, images(i).name);
    image_list{i} = imread(path);
end

processed = cell(1, length(images));

for j = 1:100:length(images)
    batch_end = min(j+99, length(images));
    % create mean reference image
    mean_img = double(image_list{j});
    tot = 1;
    for i = (j+1):batch_end
        img = double(image_list{i});
        mean_img = mean_img + img;
        tot = tot+1;
    end
    mean_img = mean_img./ tot;
    
    % subtract mean image from others
    for i = j:batch_end
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

%% Classification of images 

save_file = fullfile(image_folder_path, 'turbulence_partial_results.mat');

if isfile(save_file)
    load(save_file, 'results', 'start_frame', 'k');
    disp(['Resuming from frame ' num2str(start_frame)])
else
    results = cell(length(images), 8);
    start_frame = 1;
    k = 0;
end

i = start_frame;

while k < qty_labeled
    % display image
    img_original=processed{i};
    imshow(img_original)
    set(gcf, 'WindowState', 'maximized')
    % set(gcf, 'Color', 'k');  
    t = title(['Frame ' num2str(i)]);
    t.Color = 'black';
    
    wp = input('Turbulence? (0 = no, 1 = yes, 2 = relabel previous, 3 = throw away, 4 = exit): ');
    
    % throwaway
    if wp == 3
        results{i,2} = wp;
        results{i,1} = k;
        i = i+1;
        close;
        continue;
    
    % exit and save
    elseif wp == 4
        disp('Exiting and saving progress...');
        save(save_file, 'results', 'start_frame','k')
        close;
        break;

    % go back
    elseif wp == 2
       i = i-1;
       if results{i,2} == 3
           k = k;
       else
           k = k -1;
       end
       close;
       continue;

    % there is a wavepacket! draw a bounding box
    elseif wp == 1
        disp('Draw bounding box around the turbulence.');
        bb = drawrectangle(); 
        bbox = bb.Position; % [x y width height]
        results{i,3} = bbox(1);
        results{i,4} = bbox(2);
        results{i,5} = bbox(3); 
        results{i,6} = bbox(4);
        k = k+1;

    % there is no wavepacket
    elseif wp == 0
        results{i,3} = 'X';
        results{i,4} = 'X';
        results{i,5} = 'X'; 
        results{i,6} = 'X';
        k = k+1;

    % fail safe for invalid inputs
    else
        disp("Invalid input. Try again.")
        continue;
    end

    % store results
    results{i,1} = k;
    results{i,2} = wp;
    img = processed{i};
    [rows, cols] = size(img);
    results{i,7} = rows;
    results{i,8} = cols;
    close;
    start_frame = i + 1;
    i = i + 1;
    k
end

%% Write trained data to .txt file

labeled_images = start_frame -1;
output_file = fullfile(main_folder_path, 'turbulence_training_data.txt');
fileID = fopen(output_file, 'w');

% only perform if all images are labeled

for n = 1:labeled_images

    % do not record throwaway images
    if results{n,2} == 3
       continue;
    end
    
    % write run
    fprintf(fileID, '%d\t', results{n,1});
    % write WP ID
    fprintf(fileID, '%d\t', results{n,2});
    
    % write bounding box coords
    for j = 3:6
        item = results{n,j};
        if ischar(item) || isstring(item)
            fprintf(fileID, '%s\t', item);
        else
            fprintf(fileID, '%.0f\t', item);
        end
    end
    
    % write image size [height, width]
    fprintf(fileID, '%d\t%d\t', results{n,7}, results{n,8});
    
    %write image data
    img = processed{n};
    img_flat = reshape(img', 1, []);
    for m = 1:length(img_flat)
        pixel = img_flat(m);
        fprintf(fileID, '%.6f\t', pixel);
    end
    fprintf(fileID, '\n');
end
fclose(fileID);

% delete saved file when all data is written
if n == labeled_images && isfile(save_file)
    delete(save_file);
end

%% Save raw, labeled as .tiff to folder

outfolder = fullfile(main_folder_path,'Labeled Images'); %create this folder before running

count = 0;
total = 0;

for m = 1:labeled_images
     if results{m,2} == 3
           continue;
     end
     % if results{m,2} == 0
     %     total = total + 1;
     % end
     % if results{m,2} == 1
     %     count = 0;
     %     total = total + 1;
     % end
     j = results{m,1};
     img = image_list{m};
     imgName = sprintf('Img%06d.tiff', j);
     full_tiff_path = fullfile(outfolder, imgName);
     imwrite(img, full_tiff_path)
end

% disp('Turb percent')
% disp(count./total)