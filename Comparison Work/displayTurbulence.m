function [] = displayTurbulence(img_no, Img, image_ft,turbulentMeasure, starts, stops, h)
%% displayTurbulence(img_no, Img, image_ft, starts, stops, h)
%
%   Inputs: 
%       img_no - the current image number
%       Img - the image associated with the current image number
%       image_ft - the same image after subtracting the averaged image
%       turbulentMeasure - the statistic used to determine where turbulent
%           spots are
%       starts - the starting locations for the turbulent sections
%       stops - the ending locations for the turbulent sections
%       h - the figure to display on
%
%   This function displays the expected location of turbulent spots
%   superimposed on the image
%

% check the stops and starts
tStarts = [];
tStops = [];
tInd = 1;
for i = 1:length(stops)-1
    if stops(i) > starts(i+1)-10
        % if the stop and the next start are really close together then
        % meerge them
        if tInd > 1
            if stops(i) == tStops(tInd-1)
                % if the current stops is already in tStops then change it
                tStops(tInd-1) = stops(i+1);
            else
                % if not then make a new one
                tStarts(tInd) = starts(i);
                tStops(tInd) = stops(i+1);
                tInd = tInd+1;
            end
        else
            tStarts(tInd) = starts(i);
            tStops(tInd) = stops(i+1);
            tInd = tInd+1;
        end
    else
        if i>1
            if stops(i) ~= tStops(tInd-1)
                % if they're not close together then keep this packet as is
                tStarts(tInd) = starts(i);
                tStops(tInd) = stops(i);
                tInd = tInd+1;
                if i == length(stops)-1
                    tStarts(tInd) = starts(end);
                    tStops(tInd) = stops(end);
                end
            end
        else
            tStarts(tInd) = starts(i);
            tStops(tInd) = stops(i);
            tInd = tInd+1;
            if i == length(stops)-1
                tStarts(tInd) = starts(end);
                tStops(tInd) = stops(end);
            end
        end
    end
end
if length(starts)>1
    starts = tStarts;
    stops = tStops;
end

% turn the double Img to a three color image
figure(h)
imgHeight = size(Img,1);
image1 = zeros(imgHeight,size(Img,2),3);
temp_img = (Img-min(min(Img)))/(max(max(Img))-min(min(Img)));

% set it to a format that can be displayed
image1(:,:,1) = uint8(temp_img*255);
image1(:,:,2) = uint8(temp_img*255);
image1(:,:,3) = uint8(temp_img*255);

% do somethign with the turbulent spots
% preliminary variables
spacing = 10;
if sum(starts==0) ==0
    for i = 1:length(starts)
        shift = 0;
        for k = imgHeight:-1:1
            image1(k,starts(i)+shift:spacing:stops(i),1) = 255;
            image1(k,starts(i)+shift:spacing:stops(i),2) = floor(image1(k,starts(i)+shift:spacing:stops(i),2)/2);
            image1(k,starts(i)+shift:spacing:stops(i),3) = floor(image1(k,starts(i)+shift:spacing:stops(i),3)/2);
            image1(k,starts(i)+shift-1:spacing:stops(i)-1,1) = 255;
            image1(k,starts(i)+shift-1:spacing:stops(i)-1,2) = floor(image1(k,starts(i)+shift-1:spacing:stops(i)-1,2)/2);
            image1(k,starts(i)+shift-1:spacing:stops(i)-1,3) = floor(image1(k,starts(i)+shift-1:spacing:stops(i)-1,3)/2);
            shift = mod(shift+1,10);
        end
    end
end
% display the image and give it a title
image1 = uint8(image1);
subplot(311)
imshow(image1)
title(['Image Number: ' num2str(img_no)])


subplot(312)
% show the image after the average is subtracted.
image1 = zeros(size(image_ft,1),size(image_ft,2),3);
temp_img = (image_ft-min(min(image_ft)))/(max(max(image_ft))-min(min(image_ft)));
% temp_img = wiener2((image_ft-min(min(image_ft)))/(max(max(image_ft))-min(min(image_ft))),[5 5]);
% set it to a format that can be displayed
image1(:,:,1) = uint8(temp_img*255);
image1(:,:,2) = uint8(temp_img*255);
image1(:,:,3) = uint8(temp_img*255);
% display the image and give it a title
image1 = uint8(image1);
imshow(image1)
title('Image after subtracting average')

subplot(313)
plot(1:length(image_ft),turbulentMeasure)
xlim([0 length(image_ft)]);


%% old stuff that I'm not using anymore
% this is useful for looking at turbulence measures for the entire sequence
% subplot(221)
% plot(imageNumbers,FTIntegral,'b-',img_no,FTIntegral(img_no-imageNumbers(1)+1)...
%     ,'rx',imageNumbers,ones(1,length(imageNumbers))*mean(FTIntegral),'g-')
% xlim([imageNumbers(1) imageNumbers(end)])
% xlabel('Image Number')
% ylabel('Average Power Spectrum')
% title('Integral of Frequency Space')
% 
% % plot the number of peaks
% subplot(222)
% plot(imageNumbers,nPeaks,'b-',img_no,nPeaks(img_no-imageNumbers(1)+1),'rx',...
%     imageNumbers,ones(1,length(imageNumbers))*mean(nPeaks),'g-')
% xlim([imageNumbers(1) imageNumbers(end)])
% xlabel('Image Number')
% title('Number of Peaks')
% 
% % plot the average peak power
% subplot(223)
% plot(imageNumbers,peakAvgT,'b-',img_no,peakAvgT(img_no-imageNumbers(1)+1),...
%     'rx',imageNumbers,ones(1,length(imageNumbers))*mean(peakAvgT),'g-')
% xlim([imageNumbers(1) imageNumbers(end)])
% xlabel('Image Number')
% ylabel('Average Peak Power')
% title('Average Peak Height')
% 
% % plot the canny edge detection information
% subplot(224)
% plot(imageNumbers,cannyInfo,'b-',img_no,cannyInfo(img_no-imageNumbers(1)+1),...
%     'rx',imageNumbers,ones(1,length(imageNumbers))*mean(cannyInfo),'g-')
% xlim([imageNumbers(1) imageNumbers(end)])
% xlabel('Image Number')
% ylabel('Canny Edge Detection Measurement')
% title('Canny Edge Detection Information')

end