%% Turn video into frames for classification
clear; close all;


%% Read in and Process Images

folder_path = 'C:\UMD GRADUATE\RESEARCH\Hypersonic Image ID\videos\Test1\run34';

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
        processed{i} = rescale(img, 0, 1);
    end
end

%% Save off frames to a txt file for classification

save_file = fullfile(folder_path, 'partial_results.mat');

run = 34; %change to run number

if isfile(save_file)
    load(save_file, 'results', 'start_frame');
    disp(['Resuming from frame ' num2str(start_frame)])
else
    results = cell(length(images), 8);
    start_frame = 1;
end

output_file = fullfile(folder_path, 'video_data.txt');
fileID = fopen(output_file, 'w');

k = start_frame;
while k <= length(images)
    
    img = processed{k};

    % there are no WP's by default
    wp = 0;
    
    results{k,1} = run;
    results{k,2} = wp;
    results{k,3} = 'X';
    results{k,4} = 'X';
    results{k,5} = 'X'; 
    results{k,6} = 'X';

    % store results
    [rows, cols] = size(img);
    results{k,7} = rows;
    results{k,8} = cols;

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
    img_flat = reshape(img', 1, []);
    for m = 1:length(img_flat)
        pixel = img_flat(m);
        fprintf(fileID, '%.6f\t', pixel);
    end
    fprintf(fileID, '\n');

    k = k + 1;

    if mod(k, 100) == 0
        fprintf("\nSaved %i/%i frames", k, length(images))
    end

end
fclose(fileID);

if k == length(images) && isfile(save_file)
        delete(save_file);
end

