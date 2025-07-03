folder_path = 'C:\Users\cathe\Documents\MATLAB\trainML';
file_path = fullfile(folder_path, 'Langley_Run38_train.txt');
file_path_new = fullfile(folder_path, 'Langley_Run38_NormWL.txt');
lines = readlines(file_path);
len_lines = length(lines);

images = cell(1, len_lines);

write2 = fopen(file_path_new, 'w');

for i = 1:len_lines-1
    dat = strtrim(lines(i));
    results = strsplit(dat, '\t');

    run = "38";
    wp_id = results{2};
    bbox = results(3:6);
    rows = str2double(results{7});
    cols = str2double(results{8});

    img_data = str2double(results(9:end));
    img = reshape(img_data, cols, rows)';

    img = imcrop(img, [0, 0, 1216, 55]);
   
    [rows_crop, cols_crop] = size(img);

    scale_x = 9/23; %goal x/original x
    scale_y =  18/46; %goal y/original y

    new_rows = round(rows_crop * scale_y);
    new_cols = round(cols_crop * scale_x);
    img = imresize(img, [new_rows, new_cols]);

    images{i} = img;

    if any(strcmp(bbox, 'X'))
        % Write X X X X directly
        wp_dat = sprintf('%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t', run, wp_id, 'X', 'X', 'X', 'X', new_rows, new_cols);
    else
        % Convert and scale numeric bounding box
        bbox_num = str2double(bbox);
        bbox_num(1) = round(bbox_num(1) * scale_x);
        bbox_num(2) = round(bbox_num(2) * scale_y);
        bbox_num(3) = round(bbox_num(3) * scale_x);
        bbox_num(4) = round(bbox_num(4) * scale_y);
        wp_dat = sprintf('%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t', run, wp_id, bbox_num(1), bbox_num(2), bbox_num(3), bbox_num(4), new_rows, new_cols);
    end

    img_flat = reshape(img', 1, []);
   %wp_dat = sprintf('%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t', run, wp_id, bbox_num(1), bbox_num(2), bbox_num(3), bbox_num(4), new_rows, new_cols);

    img_data_str = sprintf('%.6f\t', img_flat);

    fprintf(write2, '%s%s\n', wp_dat, img_data_str);
end
