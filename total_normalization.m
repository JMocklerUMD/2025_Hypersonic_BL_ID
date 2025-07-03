
%%
run_num = 'Re45';  %% change run number based on data being read in
crop_rec = [0, 0, 640, 26]; %% adjust crop so that no cone/blank space is seen
BL_height = 11; %% use BL_find code with raw untrained, unprocessed images to find

folder_path = 'C:\Users\cathe\Documents\MATLAB\trainML';
file_path = fullfile(folder_path, 'CF_Re45_train.txt'); %read in file 
file_path_new = fullfile(folder_path, 'CF_Re45_normalized.txt'); %write to file 
lines = readlines(file_path);
len_lines = length(lines);
write2 = fopen(file_path_new, 'w');

WL_dat = [];
k =0;
images = cell(len_lines);

for i = 1:100
    dat = strtrim(lines(i));
    results = strsplit(dat, '\t');
    wp_id = results{2};
    rows = str2double(results{7});
    cols = str2double(results{8});
    img_data = str2double(results(9:end));
    img = reshape(img_data, cols, rows)';
    img = imcrop(img, crop_rec);

    if strcmp(wp_id, '1')
        k = k+1;
        WL_dat(k) = findwavelength(BL_height,img);   
    end
end

WL = mean(WL_dat);

for i = 1:len_lines-1
    dat = strtrim(lines(i));
    results = strsplit(dat, '\t');

    run = run_num;
    wp_id = results{2};
    bbox = results(3:6);
    rows = str2double(results{7});
    cols = str2double(results{8});

    img_data = str2double(results(9:end));
    img = reshape(img_data, cols, rows)';

    img = imcrop(img, crop_rec);

    [img, scale_x, scale_y] = scaleimage(BL_height, WL, img);

    [img, ydiff] = imagesizenormalization(BL_height, img);
    

    images{i} = img;
    [new_rows, new_cols] = size(img);

    if any(strcmp(bbox, 'X'))
        % Write X X X X directly
        wp_dat = sprintf('%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t', run, wp_id, 'X', 'X', 'X', 'X', new_rows, new_cols);
    else
        % Convert and scale numeric bounding box
        bbox_num = str2double(bbox);
        bbox_num(1) = round(bbox_num(1) * scale_x);
        bbox_num(2) = round(bbox_num(2) * scale_y + ydiff);
        bbox_num(3) = round(bbox_num(3) * scale_x);
        bbox_num(4) = round(bbox_num(4) * scale_y);
        wp_dat = sprintf('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t', run, wp_id, bbox_num(1), bbox_num(2), bbox_num(3), bbox_num(4), new_rows, new_cols);
    end

    img_flat = reshape(img', 1, []);

    img_data_str = sprintf('%.6f\t', img_flat);

    fprintf(write2, '%s%s\n', wp_dat, img_data_str);
end

fclose(write2);


%% Functions

function WL = findwavelength(BL_height, img)
    [rows, cols] = size(img);
    top_BL = rows- BL_height;
    X = img(top_BL-2,:);
    
    Fs = 1;               
    T = 1/Fs;                 
    L = cols;             
    t = (0:L-1)*T;        
    
    Y = fft(X);
    
    WL_min = 2*BL_height;
    WL_max = 3*BL_height;
    
    f = Fs*(0:L/2)/L;
    
    Y_mag = abs(Y(1:L/2+1));
    Y_mag(2:end-1) = 2 * Y_mag(2:end-1);
    
    low = find(f >= 1/WL_max, 1, 'first');
    up = find(f <= 1/WL_min, 1, 'last');
    
    [~, freq_idx_rel] = max(Y_mag(low:up));
    freq_idx = low + freq_idx_rel - 1;
    
    freq = f(freq_idx);
    WL = 1/freq;
end


function [NormIMG, ydiff] = imagesizenormalization(BL_height, img) 
    [rows, cols] = size(img);
    goal_height = 30; %goal height for images
    if rows < goal_height
        %top_height = floor((rows - BL_height)/2)
        top_block = img(1:2, :);

        rows_needed = goal_height - rows;
        reps = ceil(rows_needed / size(top_block, 1));
        full_block = repmat(top_block, reps,1);

        full_block = imnoise(full_block, 'gaussian', 0.0001);
        full_block = imgaussfilt(full_block, 0.9);

        NormIMG = cat(1, full_block, img);
        ydiff   = rows_needed; 
        % % top_height = max(floor((rows - BL_height)/2), 1);  % Top part to stretch
        % % bottom_part = img(top_height+1:end, :);
        % % 
        % % % Interpolate top part only
        % % new_top_height = goal_height - size(bottom_part, 1);
        % % [X, Y] = meshgrid(1:cols, 1:top_height);
        % % [Xq, Yq] = meshgrid(1:cols, linspace(1, top_height, new_top_height));
        % % 
        % % interp_top = interp2(X, Y, img(1:top_height, :), Xq, Yq, 'spline');
        % % interp_top = max(0, min(1, interp_top));  % Clamp to [0,1]
        % % 
        % % NormIMG = [interp_top; bottom_part];
        % % ydiff = new_top_height - top_height;



        % top_height = floor((rows - BL_height)/2);
        % 
        % top = img(1:top_height,:);
        % %top = img(1:top_height,:);
        % bott = img(top_height+1:end,:);
        % 
        % new_top = (goal_height-rows)+top_height;
        % 
        % top_stretch = imresize(top, [new_top, cols]);
        % %top_stretch = imgaussfilt(top_stretch, 1);
        % 
        % NormIMG = cat(1, top_stretch, bott);
        % 
        % ydiff = new_top-top_height;
    elseif rows > goal_height
        diff = rows - goal_height; 
        NormIMG = img(diff+1:end , :);
        ydiff = -diff;
    else
        NormIMG = img;
        ydiff = 0;
    end

end

function BL = findboundarylayer(img)
    [rows, cols] = size(img);
    
    [Gx, Gy] = gradient(img);
    EdgeMag = sqrt(Gx.^2 + Gy.^2);
    
    threshold = .08 * max(EdgeMag(:));
    
    heights = zeros(1, cols);
    for i = 1:cols
        col_data = EdgeMag(:, i); 
        high_rows = find(col_data > threshold);
        if ~isempty(high_rows)
            heights(i) = high_rows(1);  % first row from top where gradient is high
        else
            heights(i) = NaN;  % no strong edge detected
        end
    end
    
    heights = heights(~isnan(heights));  % remove empty columns
    TopBL = median(heights);  % top of boundary layer
    
    wall_row = rows;
    BL = wall_row - TopBL;
end

function [img_scale, scale_x, scale_y] = scaleimage(BL, WL, img)
    [rows, cols] = size(img);
    scale_x = 18/WL; % 19 = goal WL
    scale_y =  9/BL; % 9 = goal BL

    new_rows = round(rows * scale_y);
    new_cols = round(cols * scale_x);
    img_scale = imresize(img, [new_rows, new_cols]);
end
