% Script to invert images (some processed images were mean - img, instead of img - mean)

folder_path = 'C:\UMD GRADUATE\RESEARCH\Hypersonic Image ID\videos\Test1\ConeFlare_Shot67_re45_0deg\';
file_path = fullfile(folder_path, 'training_data_CF_Re45_FINAL.txt');

lines = readlines(file_path);
len_lines = length(lines);

inverted_images = cell(1, len_lines);

writeto = fopen(file_path, 'w');

for i = 1:len_lines-1
    dat = strtrim(lines(i));
    results = strsplit(dat, '\t');

    run = results{1};
    wp_id = results{2};
    bbox = results(3:6);
    rows = str2double(results{7});
    cols = str2double(results{8});

    img_data = str2double(results(9:end));
    img = reshape(img_data, cols, rows)';

    img_inv = 1 - img;

    img_flat = reshape(img_inv', 1, []);

    wp_dat = sprintf('%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t', run, wp_id, bbox{1}, bbox{2}, bbox{3}, bbox{4}, rows, cols);
    inv_img_data = sprintf('%.6f\t', img_flat);

    fprintf(writeto, '%s%s\n', wp_dat, inv_img_data);
end

fclose(writeto);
