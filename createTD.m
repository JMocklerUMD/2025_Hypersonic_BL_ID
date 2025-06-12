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
        img = mean_img - img;
        processed{i} = rescale(img, 0, 1);
    end
end

%%
save_file = fullfile(folder_path, 'partial_results.mat');

if isfile(save_file)

    load(save_file, 'results', 'start_frame');
    disp(['Resuming from frame ' num2str(start_frame)])
else
    results = cell(length(images), 8);
    start_frame = 1;
end

i = start_frame;
while i <= length(images)
    figure;
    imshow(processed{i});
    set(gcf, 'WindowState', 'maximized')
    title(['Frame ' num2str(i)]);
    
    wp = input('Wave packet? (0 = no, 1 = yes, 2 = relabel previous, 3 = throw away, 4 = exit): ');
    
    if wp == 3
        results{i,2} = wp;
        i = i+1;
        close;
        continue;
    end

    if wp == 4
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

    start_frame = i + 1;

    i = i +1;
end

output_file = fullfile(folder_path, 'wavepacket_labels.txt');
fileID = fopen(output_file, 'w');
%%

if i == length(images) + 1
    for i = 1:length(images)
         if results{i,2} == 3
            continue;
         end

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
end