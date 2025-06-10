folder_path = 'C:\Users\Joseph Mockler\Documents\test_imgs';

% read in files in folder
images = dir(fullfile(folder_path, '*.tif'));

% read in image data 
image_list = cell(1, length(images));
for i = 1:length(images)
    path = fullfile(folder_path, images(i).name);
    image_list{i} = imread(path);
end

% create mean reference image
mean_img = double(image_list{1});
for i = 2:length(image_list)
    img = double(image_list{i});
    mean_img = mean_img + img;
end
mean_img = mean_img / length(image_list);

% subtract mean image from others
processed = cell(1, length(images));
for i = 1:length(images)
    img = double(image_list{i});
    img = mean_img - img;
    img = rescale(img, 0, 1);
    processed{i} = imadjust(imgaussfilt(img, 1));
end
%%

if isfile(save_file)
    load(save_file, 'results', 'start_frame');
    disp(['Resuming from frame ' num2str(start_frame)])
else
    save_file = fullfile(folder_path, 'partial_results.mat');
    results = cell(length(images), 8);
    start_frame = 1;
end

i = start_frame;
while i <= length(images)
    figure;
    imshow(processed{i});
    title(['Frame ' num2str(i)]);
    
    wp = input('Wave packet? (0 = no, 1 = yes, 2 = relabel previous, 3 = exit): ');
    
    if wp == 3
        disp('Exiting and saving progress...');
        save(save_file, 'results', 'start_frame')
        close;
        break;
    end

    if wp == 2
       i = i-1;
       close;
       continue;
    end

    results{i,1} = i;
    results{i,2} = wp;

    if wp == 1
        disp('Draw bounding box around the wave packet.');
        bb = drawrectangle(); 
        bbox = bb.Position; % [x y width height]
        results{i,3} = bbox(1);
        results{i,4} = bbox(2);
        results{i,5} = bbox(3); 
        results{i,6} = bbox(4);
    else
        results{i,3} = 'X';
        results{i,4} = 'X';
        results{i,5} = 'X'; 
        results{i,6} = 'X';
    end
    img = processed{i};
    [rows, cols] = size(img);
    results{i, 7} = rows;
    results{i,8} = cols;
    close;

    start_frame = i +1;
    save(save_file, 'results', 'start_frame');

    i = i +1;
end

output_file = fullfile(folder_path, 'wavepacket_labels.txt');
fileID = fopen(output_file, 'w');


for i = 1:length(images)
    fprintf(fileID, '%d\t', results{i,1});
    fprintf(fileID, '%d\t', results{i,2});
    
    for j = 3:6
        item = results{i,j};
        if ischar(item) || isstring(item)
            fprintf(fileID, '%s\t', item);
        else
            fprintf(fileID, '%.0f\t', item);
        end
    end
    
    fprintf(fileID, '%d\t%d\t', results{i,7}, results{i,8});

    img = processed{i};
    img_flat = reshape(img', 1, []);
    for k = 1:length(img_flat)
        pixel = img_flat(k);
        fprintf(fileID, '%.6f\t', pixel);
    end
    fprintf(fileID, '\n');
end
fclose(fileID);

if start_frame > length(images) && isfile(save_file)
    delete(save_file);
end
