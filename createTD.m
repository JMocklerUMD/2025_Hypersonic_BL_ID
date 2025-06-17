%% Create Training Data

%% Read in and Process Images

folder_path = 'C:\UMD GRADUATE\RESEARCH\Hypersonic Image ID\videos\Test1\ConeFlare_Shot64_re33_0deg';

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
        img = mean_img - img;
        processed{i} = rescale(img, 0, 1);
    end
end

%% Manually train data 

save_file = fullfile(folder_path, 'partial_results.mat');

run = 4120; %change to run number

if isfile(save_file)

    load(save_file, 'results', 'start_frame');
    disp(['Resuming from frame ' num2str(start_frame)])
else
    results = cell(length(images), 8);
    start_frame = 1;
end

i = start_frame;
while i <= length(images)
    
    % display image
    img = processed{i};
    %img_enhanced = imadjust(img);
    %imshow(img_enhanced)
    imshow(img)
    set(gcf, 'WindowState', 'maximized')
    set(gcf, 'Color', 'k');  
    t = title(['Frame ' num2str(i)]);
    t.Color = 'white';
    
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
    results{i,1} = run;
    results{i,2} = wp;
    img = processed{i};
    [rows, cols] = size(img);
    results{i, 7} = rows;
    results{i,8} = cols;
    close;
    start_frame = i + 1;
    i = i + 1;
end

%% Write trained data to .txt file

output_file = fullfile(folder_path, 'training_data.txt');
fileID = fopen(output_file, 'w');

% only perform if all images are labeled
if i == length(images) + 1 % remove if statement if training stopped early and run section
    for k = 1:length(images)

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
        img = processed{k};
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