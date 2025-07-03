folder_path = 'C:\Users\cathe\Documents\MATLAB\trainML';
file_path = fullfile(folder_path, 'CF_Re33_train.txt');
file_path_new = fullfile(folder_path, 'Langley_Run34_cropped.txt');
lines = readlines(file_path);
len_lines = length(lines);

images = cell(1, len_lines);

write2 = fopen(file_path_new, 'w');

for i = 1:len_lines-1
    dat = strtrim(lines(i));
    results = strsplit(dat, '\t');

    run = "33";
    wp_id = results{2};
    bbox = results(3:6);
    rows = str2double(results{7});
    cols = str2double(results{8});

    img_data = str2double(results(9:end));
    img = reshape(img_data, cols, rows)';

    img = imcrop(img, [0, 0, 1200, 55]); % change these depending on crop
    img = NormBLdown(17, img); % change forst input based on BL thickness

    [new_rows, new_cols] = size(img);

    img_flat = reshape(img', 1, []);

    wp_dat = sprintf('%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t', run, wp_id, bbox{1}, bbox{2}, bbox{3}, bbox{4}, new_rows, new_cols);

    img_data_str = sprintf('%.6f\t', img_flat);

    fprintf(write2, '%s%s\n', wp_dat, img_data_str);
end

function NormIMG = NormBLdown(BL_height, img) 
    [rows, cols] = size(img);

    top_height = floor((rows - BL_height)/2);
    
    top = img(1:top_height,:);
    bott = img(top_height+1:end,:);
    
    bott_height = size(bott,1);
    
    new_height = round(BL_height/0.3);
    
    new_top = new_height - bott_height;
    
    top_stretch = imresize(top, [new_top, cols]);
    
    NormIMG = cat(1, top_stretch, bott);
end



