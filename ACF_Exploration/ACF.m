%import data from file

clear %clear variables to start fresh from last test - can remove later

fileName = "training_data_explicit.txt";

file = readtable(fileName); %read in txt file data as table

%in the future perhaps split up data by test source (file.Var1)

%write data from table to relevent arrays
yesNo2Mode = categorical(file.Var2); %categorical array of 'Yes' or 'No' for if image has a second-mode wave or not
x = int16(file.Var3); %store bounding box values (x, y, length, height) as int16 to reduce memory burden
y = int16(file.Var4); %if bounding box doesn't exist 'X' from data file gets changed to 0
length = int16(file.Var5);
height = int16(file.Var6);
bbox = [int16(file.Var3) int16(file.Var4) int16(file.Var5) int16(file.Var6)];
pixelLength = int16(file.Var7); %image metadata
pixelHeight = int16(file.Var8); %image metadata
image_data_table = file(:,9:size(file,2));
image_data = table2array(image_data_table);

%takes about 2 minutes to run here for Intel i7 on an HP laptop purchased
%in 2024


%code to create .tiff files
%probably loses some data
%only works up to img2069 because image size decreases after that
%{
for i = 1:size(file,1)
%funtion to map onto uint16 values (thanks ChatGPT among other things)
imgMatrix = reshape(image_data(i,:), [pixelHeight(i), pixelLength(i)]);
imgMatrix = imgMatrix';
[min,max] = bounds(imgMatrix,"all");
imgMatrix = maprange(imgMatrix,min,max);
imgMatrix = uint16(imgMatrix);
%imshow(imgMatrix,[])
str = "C:\Users\tyler\Documents\GitHub\2025_Hypersonic_BL_ID\ACF_Exploration\tiffImages\" + sprintf("img%d.tiff",i);
imwrite(imgMatrix,str);
%write to preexisting folder
end
%}

%create ground truth: images with bounding boxes and labels
tiffDatastore = imageDatastore('C:\Users\tyler\Documents\GitHub\2025_Hypersonic_BL_ID\ACF_Exploration\tiffImages\');
dataSource = groundTruthDataSource(tiffDatastore);
ldc = labelDefinitionCreator();
addLabel(ldc,'secondModeWave',labelType.Rectangle);
labelNames = {'secondModeWave'};
secondModeWaveTruth = zeros(2069,4);
for i = 1:2069
        secondModeWaveTruth(i,:) = [bbox(i,:)];
end

labelData = table(secondModeWaveTruth,'VariableNames',labelNames);
labelDefs = create(ldc);
gTruth = groundTruth(dataSource,labelDefs,labelData);

%list of training images
[trainingImgList,bboxLabels] = objectDetectorTrainingData(gTruth,SamplingFactor=2); %pulls every 2 images
trainingImgWithLabels = combine(trainingImgList,bboxLabels);

%create aggregate channel feature (ACF) object detector
acf = trainACFObjectDetector(trainingImgWithLabels);
img1 = read(tiffDatastore); %read function reads sequentially from datastore
[predictBbox,confidence] = detect(acf,img1); 
img1Ann = insertObjectAnnotation(img1,'rectangle',predictBbox,confidence);
img1Both = insertObjectAnnotation(img1Ann,'rectangle',bbox(1,:),'manual');


%files = imList.Files
%inside objectDetectorTrainingData: objectDetectorTrainingData(gTruth,SamplingFactor=10,NamePrefix="turtleFrame",WriteLocation="trainingImages"
% im = imread(files{1});
% bb = boxLabels.LabelData{1,1}
% im = insertObjectAnnotation(im, ...
%     "rectangle",bb,"Turtle");
% imshow(im)

