% Image pre-processor for ResNet (For Training - With Labeling)

% Define the folder where the images are
imageFolder = 'C:\Users\Christoph Neisess\Documents\University Things\UMD 2018-2019\ENAE398H\run_4017';
files = dir(fullfile(imageFolder, '*.tif'));

% Read in images
for k = 1:numel(files)
    imfile = fullfile(imageFolder,files(k).name);
    imgArray{k} = double(imread(imfile));
end

% Choose the image to start on in the set (if picking up from a previous
% stopping point, otherwise enter 1)
prompt = 'Enter the number of the starting image: ';
startImg = input(prompt);
fprintf('\n');

% Create a reference image

for m = floor(startImg/100):floor(length(imgArray)/100)
    if (m*100)+100 > length(medArray)
        endimg = length(medArray);
    else
        endimg = (m*100)+100;
    end
    sum = imgArray{(m*100)+1};
    for n = (m*100)+2:endimg
        sum = sum + imgArray{n};
        avg = sum./100;
    end
    for n = (m*100)+1:endimg
        if n <= length(medArray)
            subImage{n} = imgArray{n} - avg; 
        end
    end
end
    
%Create a length to reference
listsize = size(subImage);
numfiles = listsize(2);

% Do some tweaking on the images
for k = startImg:numfiles
    % Normalize each image's pixel values to be between 0 and 1
    subImage{k} = (subImage{k} - min(min(subImage{k})))/(max(max(subImage{k})) -min(min(subImage{k})));

    % (Optional): Adjust the spread of pixel values to boost contrast if
    % the images are still murky after mean-subtraction
    %adjImage{k} = imadjust(subImage{k},stretchlim(subImage{k})[]);

end

% Classify each image and save it with a new filename
for k = startImg:numfiles
    
    % Display the image in question
    imshow(subImage{k})
    
    % Enter whether it has second mode waves or not
    prompt = 'Second mode wave present? (0=yes, 1=no) ';
    x = input(prompt);
    fprintf('\n');
    
    if x == 0
        imgName = sprintf('Processed_SMW_Present_%d',k);
    else
        imgName = sprintf('Processed_SMW_Not_Present_%d',k);
    end
    
    % Add the coordinates of the bounding box
    if x == 0
        fprintf('Click the top left and bottom right corners of the SMW\n\n');
        [px,py] = ginput(2);
        bboxc = [px(1) py(1) (px(2)-px(1)) (py(2)-py(1))];
        bboxc = uint16(bboxc);
        FID = fopen('Processed_Predictions_Bounding_Boxes.txt','a');
        fprintf(FID, '%u %u %u %u\r\n', bboxc);
        fclose(FID);
    else
        bboxc = uint16([0 0 0 0]);
        FID = fopen('Processed_Predictions_Bounding_Boxes.txt','a');
        fprintf(FID, '%u %u %u %u\r\n', bboxc);
        fclose(FID);
    end

    % Save the image with the new label
    fileName = fullfile('C:\Users\Christoph Neisess\Documents\University Things\UMD 2018-2019\ENAE398H\Processed_Predictions\',[imgName,'.tif']);
    imwrite(subImage{k},fileName);  
    
    close all;
end
