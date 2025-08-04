% Script to determine accuracy of automated classification methods

main_folder_path = 'C:\Users\Joseph Mockler\Documents\GitHub\2025_Hypersonic_BL_ID';
%Folder containing raw .tif images corresponding to labeled data in text file
image_folder_path = 'C:\Users\Joseph Mockler\Documents\GitHub\2025_Hypersonic_BL_ID\Final Images'; 
training_data_file = fullfile(main_folder_path, 'filtered_training_data.txt');
algorithmOutput_ID = MUSIC_ID; % Change based on method
algorithmOutput_locs = MUSIC_locs; % Change based on method
print_flag = 0; % Set to 0 if you do not want to plot results

images = dir(fullfile(image_folder_path, '*.tif')); 
image_list = cell(1, length(images));
for i = 1:length(images)
    path = fullfile(image_folder_path, images(i).name);
    image_list{i} = imread(path);
end

num_images = length(algorithmOutput_ID);
fileID = fopen(training_data_file, "r");

training_data = {};
line_num = 0;

while feof(fileID) ~= 1
    line = fgetl(fileID);
    line_num = line_num +1;
    dat  = strsplit(line, '\t');
     
    frame = str2double(dat{1});
    WP_ID = str2double(dat{2});

    bbox_dat = dat(3:6);

    if strcmp(bbox_dat(1), 'X')
        bbox = ['X', 'X', 'X', 'X'];
    else
        bbox = bbox_dat;
    end

    rows = str2double(dat{7});
    cols = str2double(dat{8});
    img_size = [rows, cols];

    img_dat = str2double(dat(9:end-1));

    training_data{line_num, 1} = frame;
    training_data{line_num, 2} = WP_ID;
    training_data{line_num, 3} = bbox;
    training_data{line_num, 4} = img_size;
    training_data{line_num, 5} = img_dat;
end

TruePositive = [];
FalseNegative = [];
FalsePositive = [];
TrueNegative = [];

for i = 1:num_images
    wp = training_data{i, 2};
    bbox = training_data{i, 3};
    img_size = training_data{i, 4};
    rows = img_size(1);
    cols = img_size(2);
    image = double(image_list{i});
    image = rescale(image, 0,1);
    false_neg = 0;
    false_pos = 0;
    true_pos = 0;
    true_neg = 0;

    % First establish the bounds on the classification
    if algorithmOutput_ID(i) == 1 % positive WP classification
        bbox_start = algorithmOutput_locs(1, i);
        bbox_end = algorithmOutput_locs(2, i);
    else
        bbox_start = 0;
        bbox_end = 0;
    end

    if wp == 1
        x_start = str2double(bbox{1});
        x_end = str2double(bbox{3}) + x_start;
    else
        x_start = 0;
        x_end = 0;
    end
    
    % Now go along the streamline and check if the classification at each column in the frame
    for j = 1:cols
        % If j is within the autoclassification bounds...
        if (bbox_start < j && j < bbox_end)
            auto_class = 1;
        else
            auto_class = 0;
        end

        % If j is within the human classification bounds...
        if (x_start < j && j < x_end)
            human_class = 1;
        else
            human_class = 0;
        end

        % Now perform the logic to check if it got it right or not
        if auto_class == 1
            if human_class == 1
                true_pos = true_pos + 1;
            elseif human_class == 0
                false_pos = false_pos + 1;
            end
        elseif auto_class == 0
            if human_class == 1
                false_neg = false_neg + 1;
            elseif human_class == 0
                true_neg = true_neg + 1;
            end

        end
    end

    % Normalize to get % of the frame
    TrueNegative(i) = true_neg/cols;
    TruePositive(i) = true_pos/cols;
    FalsePositive(i) = false_pos/cols;
    FalseNegative(i) = false_neg/cols;

    % Visualize
    if print_flag == 1
    figure;
    imshow(image);
    hold on
    if algorithmOutput_ID(i) == 1
        xline(bbox_start, 'r', 'Linewidth', 1.5, 'Label', "MUSIC start")
        xline(bbox_end, 'r', 'Linewidth', 1.5, 'Label', "MUSIC end")
    end
    if wp == 1
        x = str2double(bbox{1});
        y = str2double(bbox{2});
        width = str2double(bbox{3});
        height = str2double(bbox{4});
        rectangle('Position', [x, y, width, height],'EdgeColor', 'b', 'Linewidth', 1.5)
    end
    end
end

setTruePositive = mean(TruePositive);
setTrueNegative = mean(TrueNegative);
setFalsePositive = mean(FalsePositive);
setFalseNegative = mean(FalseNegative);

fprintf('Whole Set Accuracy: %.4f%%\n', 100*(setTruePositive+setTrueNegative));
fprintf('True Positive Rate: %.4f%%\n', 100*setTruePositive);
fprintf('True Negative Rate: %.4f%%\n', 100*setTrueNegative);
fprintf('False Positive Rate: %.4f%%\n', 100*setFalsePositive);
fprintf('False Negative Rate: %.4f%%\n', 100*setFalseNegative);