function [] = displayPackets(Img, img_no, dx, wave_packets, cone, BL_height, ...
    starts, stops, f, FT, filtered_img, filtered_img1, image_ft,h)
% displayPackets(Img, img_no, dx, wave_packets, cone, BL_height, ...
%     starts, stops, f, FT, filtered_img, filtered_img1, image_ft,h)
%
%   Inputs:
%       Img - the origional image that the information is derived from
%       img_no - the number of this image in the sequence
%       dx - the length associated with a step of 1 pixel
%       wave_packets - the output from findWavePackets
%       cone - the row where the cone stops intruding into the image
%       BL_height - the highest row in the image where the boundary layer
%           exists
%       starts - the starting location for any wave packets in the image
%       stops - the stopping location for any wave packets in the image
%       f - the frequencies used in the fourier transform
%       FT - the output from said fourier transform
%       filtered_img - the image filtered for 2nd mode wave packets
%       filtered_img1 - the image filtered for 1st harmonic waves
%       image_ft - the image after subtracting the averaged image
%       h - the window to display anything in
%
%   This function displays information found when determining the location
%   and extent of 2nd mode wave packets.
%

global rowShift

% some supporting calculations

BL_thickness = cone-BL_height;
% Frequency bounds for 2nd mode frequency
find_min = find(f>1/(4.5*BL_thickness),1,'first');
find_max = find(f<(1/(BL_thickness*1.5)),1,'last');

% maximum row
[row,~] = find(FT(:,find_min:find_max) == max(max(FT(:,find_min:find_max))));
% if there is no FT
if isempty(row)
    row = 1;
end
% shift to the one that should be currently on display
displayRow = row(1)+rowShift;
if displayRow <1
    displayRow = 1;
elseif displayRow > size(filtered_img,1)
    displayRow = size(filtered_img,1);
end
% display
% convert the greyscale image into a color image so that the BL 
% can be outlined in color
figure(h)
subplot(311)
image1 = zeros(size(Img,1),size(Img,2),3);
temp_img = (Img-min(min(Img)))/(max(max(Img))-min(min(Img)));

% set it to a format that can be displayed
image1(:,:,1) = uint8(temp_img*255);
image1(:,:,2) = uint8(temp_img*255);
image1(:,:,3) = uint8(temp_img*255);

% outline the BL
image1([cone,BL_height],:,1) = 255;
image1([cone,BL_height],:,2) = 0;
image1([cone,BL_height],:,3) = 0;

% mark the wave packets
% this section marks the starts from each row in blue and the stops from
% each row in purple then it puts the paired starts in green and the paired
% stops in pink
if ~isempty(wave_packets)
for i = 1:size(wave_packets,1)
    for j = 1:size(wave_packets,2)
        for k = 1:size(wave_packets,3)
            if wave_packets(i,j,k) ~= 0
                if k == 1
                    if i == displayRow
                        % if the packet is in the row currently being
                        % displayed then use a different set of colors
                        image1(1:i+BL_height-1,wave_packets(i,j,k),1) = 100;
                        image1(1:i+BL_height-1,wave_packets(i,j,k),2) = 200;
                        image1(1:i+BL_height-1,wave_packets(i,j,k),3) = 50;
                    else
                        image1(1:i+BL_height-1,wave_packets(i,j,k),1) = 0;
                        image1(1:i+BL_height-1,wave_packets(i,j,k),2) = 100;
                        image1(1:i+BL_height-1,wave_packets(i,j,k),3) = 255;
                    end
                else
                    if i == displayRow
                        % if the packet is in the row currently being
                        % displayed then use a different set of colors
                        image1(1:i+BL_height-1,wave_packets(i,j,k),1) = 255;
                        image1(1:i+BL_height-1,wave_packets(i,j,k),2) = 100;
                        image1(1:i+BL_height-1,wave_packets(i,j,k),3) = 0;
                    else
                        image1(1:i+BL_height-1,wave_packets(i,j,k),1) = 100;
                        image1(1:i+BL_height-1,wave_packets(i,j,k),2) = 0;
                        image1(1:i+BL_height-1,wave_packets(i,j,k),3) = 100;
                    end
                end
            end
        end
    end
end
end

% Place the resulting bounds
if ~isempty(starts)
    for i = 1:length(starts)
        image1(:,floor(starts(i)),1) = 0;
        image1(:,floor(starts(i)),2) = 255;
        image1(:,floor(starts(i)),3) = 0;
    end
end
if ~isempty(stops)
    for i = 1:length(stops)
        image1(:,ceil(stops(i)),1) = 255;
        image1(:,ceil(stops(i)),2) = 0;
        image1(:,ceil(stops(i)),3) = 255;
    end
end

% display the image and give it a title
image1 = uint8(image1);
imshow(image1)
title(['Image Number: ' num2str(img_no) '  BL Thickness: ' num2str(cone-BL_height,2)])

subplot(313)
% plot the filtered and unfiltered intensity profiles

% get the x locations (Only for 1296 and 1297)
% xs = 585.62:(767.4-585.62)/(length(image_ft)-1):767.4;

% Use the column numbers rather than the exact location
xs = 1:length(image_ft);
plot(xs,filtered_img(displayRow,:),'k-',...  %40-12-BL_height+1
    xs,(image_ft(displayRow+BL_height-1,:)),'c-',...
    xs,filtered_img1(displayRow,:),'r-')
hold on
% outline the wave_packets location for the current display row
ys = ylim;
ys = ys(1):ys(2);
for j = 1:size(wave_packets,2)
    if displayRow <= size(wave_packets,1)
        if wave_packets(displayRow,j,1) ~= 0
            plot(ones(size(ys))*wave_packets(displayRow,j,1),ys,'g--',...
                ones(size(ys))*wave_packets(displayRow,j,2),ys,'m--')
        end
    end
end
hold off
% limit the axis and label things
xlim([min(xs) max(xs)])
xlabel('Column (# of pixels from left side of image)')
legend('Filtered 2nd mode','Unfiltered','Filtered 1st Harmonic')
if displayRow == row(1);
    isMax = '  MAX Frequency Power';
else
    isMax = [];
end
title(['Intensity Profiles for y = ' num2str(1000*dx*(cone-(displayRow+BL_height-1)),3) 'mm above the cone' isMax])

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

% Code for plotting the fourier transform results

% % plotting 1st mode frequencies
% % plot the frequency domain for the range of detection
% % find the frequency bounds to use (centered on 1st harmonic)
% find_min1 = find(f>1/(2*BL_thickness),1,'first');
% find_max1 = find(f<(1/(BL_thickness*1)),1,'last');
% plot(f(find_min1:find_max1),FT(:,find_min1:find_max1))
% ylim([1 10^7]);

% grid on
% % plotting 2nd mode frequencies (1/mm)
% plot(1/dx*f(find_min:find_max),FT(:,find_min:find_max))
% xlim([1/dx*f(find_min) 1/dx*f(find_max)])

% % plotting 2nd mode frequencies (1/pixel)
% plot(f(find_min:find_max),FT(:,find_min:find_max));
% xlim([f(find_min) f(find_max)])
% 
% title(['1st harmonic frequency: ' num2str(1/(1.5*(BL_thickness+0))) '     2nd mode frequency: ' ...
%     num2str(1/(3*(BL_thickness+0)))])
% % xlabel('Frequency [1/mm]')
% xlabel('frequency 1/pixel')
% ylabel('Power |FT|^2')
end