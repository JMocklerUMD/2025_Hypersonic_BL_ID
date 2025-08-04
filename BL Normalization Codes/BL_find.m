%folder of raw .tif images
image_folder_path = "C:\Users\tyler\Desktop\NSSSIP25\Machine Learning Classification - NSSSIP25\Example Data and Outputs\LangleyRun34_105_116ms_rawTIFimages"; 

% read in files in folder
images = dir(fullfile(image_folder_path, '*.tif'));

% read in image data 
image_list = cell(1, length(images));
for i = 1:length(images)
    path = fullfile(image_folder_path, images(i).name);
    image_list{i} = double(imread(path));
end

tot = 1;
mean_img = image_list{1};
for i = 2:length(images)
    img = image_list{i};
    mean_img = mean_img + img;
    tot = tot+1;
end
mean_img = mean_img./ tot;
mean_img = rescale(mean_img, 0, 1);
figure;
histogram(mean_img)
figure;
imshow(mean_img)

%%
threshold_scale = 0.12; % gradient threshold, change depending on image set
dark_cone = 1; % is the cone darker than the rest of the mean image?

%%
img = mean_img;

[rows, cols] = size(img);

[Gx, Gy] = gradient(img);
EdgeMag = sqrt(Gx.^2 + Gy.^2);

threshold = threshold_scale * max(EdgeMag(:));

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

heights = heights(~isnan(heights));

row_intensities = median(img, 2);
row_gradient = gradient(row_intensities);

if dark_cone == 0
    [~, cone_height_top] = max(row_gradient);
else
    [~, cone_height_top] = min(row_gradient);
end

cone_height = rows - cone_height_top;
TopBL = median(heights);
delta = (rows-cone_height) - TopBL;

fprintf('Top of boundary layer: %.2f (row)\n', TopBL);
fprintf('Cone height: %.2f pixels\n', cone_height);
fprintf('True Boundary layer thickness: %.2f pixels\n', delta);

figure;
imshow(mean_img, []);
hold on;
yline(TopBL, 'r--', 'LineWidth', 1.5);
yline(cone_height_top, 'r', 'LineWidth', 1.5);



