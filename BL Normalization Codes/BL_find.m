
folder_path = 'C:\Users\cathe\Documents\MATLAB\trainML\Re33\Re33';

% read in files in folder
images = dir(fullfile(folder_path, '*.tif'));

% read in image data 
image_list = cell(1, length(images));
for i = 1:length(images)
    path = fullfile(folder_path, images(i).name);
    image_list{i} = imread(path);
end

mean_img = double(image_list{1});

for k = 2:length(images)
    img = double(image_list{k});
    mean_img = mean_img + img;
end

mean_img = mean_img./ length(images);

mean_img = rescale(mean_img, 0, 1);
figure;
histogram(mean_img)
figure;
imshow(mean_img)

%%

img = mean_img;

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
delta = wall_row - TopBL;  % boundary layer thickness

fprintf('Top of boundary layer: %.2f (row)\n', TopBL);
fprintf('Boundary layer thickness: %.2f pixels\n', delta);

figure;
imshow(mean_img, []);
hold on;
yline(TopBL, 'r', 'LineWidth', 1.5);


