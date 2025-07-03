%% Cone Flare Splitting Script
clc; clear variables

% read in files in folder
folder_path = 'C:\Users\rclat\OneDrive\Documents\re33_10deg';
images = dir(fullfile(folder_path, '*.tif'));

% read in image data 
image_list = cell(1, length(images));
for i = 1:length(images)
    path = fullfile(folder_path, images(i).name);
    image_list{i} = imread(path);
end

% find corner of cone flare
tot=0;
x=0;
sigma=4;
for i = 1:length(images)
    ref_img = image_list{i};    
    % Gaussian blur and canny edge detection
    blurred = imgaussfilt(ref_img,sigma);
    edges = edge(blurred,'Canny');
    % Hough transform to get lines
    [H,T,R] = hough(edges);
    P = houghpeaks(H,5,'threshold',ceil(0.3 * max(H(:))));
    lines = houghlines(edges,T,R,P,'FillGap',20,'MinLength',50);
    figure(1), imshow(ref_img), hold on
    max_len = 0;
    for k = 1:length(lines)
       xy = [lines(k).point1;lines(k).point2];
       plot(xy(:,1),xy(:,2),'Color','green');
    end
    if length(lines) >= 2
        % Get line endpoints
        line1 = lines(1);
        line2 = lines(2);
        % Convert lines to general form: Ax + By = C
        [A1, B1, C1] = line_to_abc(line1.point1, line1.point2);
        [A2, B2, C2] = line_to_abc(line2.point1, line2.point2);
        % Solve for intersection
        M = [A1 B1; A2 B2];
        b = [C1; C2];    
        if det(M) ~= 0
            intersection = M\b;
            xline(intersection(1),'r')
        else
            disp('Lines are parallel — no intersection found.')
        end
    end
    % Throw out outlier values that are far from actual corner
    if i==1
        base=intersection(1);
    end
    if intersection(1)<base+10 && intersection(1)>base-10
        tot=tot+intersection(1);
    else
        x=x+1;
    end
    hold off
end
% Print average corner location (pixels in x direction)
avg=tot/(length(images)-x)

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
    img = img - mean_img;
    [rows, cols] = size(img);
    factor = avg/cols;
    split_column = round(cols * factor);   
    img1 = img(:, 1:split_column);      % Left part
    img2 = img(:, split_column+1:end);  % Right part
    processed{i}{1} = rescale(img1, 0, 1);
    processed{i}{2} = rescale(img2, 0, 1);
end

figure(2)
imshow(processed{100}{1})
figure(3)
imshow(processed{100}{2})

% ---------Functions------------

function [A, B, C] = line_to_abc(p1, p2)
    % Converts two points into general line form Ax + By = C
    A = p2(2) - p1(2);
    B = p1(1) - p2(1);
    C = A*p1(1) + B*p1(2);
end