function displayWavelet(h,Img,image_ft,img_no,cone,BL_height,wave_packets,starts,stops,wavData1,cmax1,waveScales,varargin)
%% displayWavelet(h,Img,image_ft,img_no,cone,BL_height,wave_packets,starts,stops,wavData1,cmax1,waveScales,[wavData2,cmax2,...,turbulentSpots])
%
%   Inputs:
%       h - the figure to use for plotting
%       Img - the origional image
%       image_ft - the image after subtracting an averaged image
%       wavData1 - the wavelet transform magnitude results for each row of
%           the boundary layer. wavData1 should be length(waveScales) X
%           imgLength X BL_thickness+1
%       cmax1 - the maximum expected value for wavData1
%       waveScales - the scales used for the wavelet analysis, to become
%           the y axis
%       wavData2,cmax2,... - additional individual wavelet analysis results
%       turbulentSpots - the start and stop locations of the turbulent
%           spots
%
%
global rowShift
opengl software
clf(h)
scaleTo255 = @(img) uint8(255*(img-min(min(img)))/(max(max(img))-min(min(img))));
yLimits =  [min(waveScales) max(waveScales)];
xLimits = [1 length(Img)];
imageSpace = size(Img,1)/(8*40);
BL_thickness = cone-BL_height;
displayRow = BL_thickness+1+rowShift;
if displayRow < 1
    displayRow = 1;
    rowShift = -BL_thickness;
%     disp('Hi')
elseif displayRow > BL_thickness+1
    displayRow = BL_thickness+1;
    rowShift = 0;
end

% put in wave packet indicators
image1 = zeros([size(Img),3]);
image1(:,:,1) = scaleTo255(Img);
image1(:,:,2) = scaleTo255(Img);
image1(:,:,3) = scaleTo255(Img);
% outline the BL
image1([cone,BL_height],:,1) = 255;
image1([cone,BL_height],:,2) = 0;
image1([cone,BL_height],:,3) = 0;

% mark the wave packets
% this section marks the starts from each row in blue and the stops from
% each row in purple then it puts the paired starts in green and the paired
% stops in pink
if ~isempty(wave_packets)
    wave_packets(:,:,1) = floor(wave_packets(:,:,1));
    wave_packets(:,:,2) = ceil(wave_packets(:,:,2));
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


figure(h)
if ~isempty(varargin)
    numExtraData = floor(length(varargin)/2);
    if mod(length(varargin),2) == 1
        turbulentSpots = varargin{end};
        for j = 1:size(turbulentSpots,1)
            if turbulentSpots(j,1) ~= 0
                image1(:,floor(turbulentSpots(j,1)),1) = 255;
%                 image1(:,floor(turbulentSpots(j,1)),2:3) = 0;
                image1(:,floor(turbulentSpots(j,2)),3) = 255;
%                 image1(:,floor(turbulentSpots(j,2)),2:3) = 0;
            end
        end
    end
    axes('OuterPosition',[0 1-imageSpace 1 imageSpace]);
    imshow(image1./255)
    title(['Image Number ' num2str(img_no)]);
    axes('OuterPosition',[0 1-2*imageSpace 1 imageSpace]);
    image1 = zeros([size(image_ft),3]);
    image1(:,:,1) = scaleTo255(image_ft);
    image1(:,:,2) = scaleTo255(image_ft);
    image1(:,:,3) = scaleTo255(image_ft);
    imshow(image1./255)
    if numExtraData == 1 || numExtraData == 3
        if numExtraData == 1
            axes('OuterPosition',[0 0 1/2 1-2*imageSpace]);
            surf(1:length(Img),waveScales,wavData1(:,:,end+rowShift),'EdgeColor','none','FaceColor','interp')
            xlim(xLimits);
            ylim(yLimits);
            view(0,90);
            caxis([0 cmax1]);
            title(['Wavelet transform data for ' num2str(-rowShift) ' rows above the cone'])
            axes('OuterPosition',[1/2 0 1/2 1-2*imageSpace]);
            surf(1:length(Img),waveScales,varargin{1},'EdgeColor','none','FaceColor','interp');
            xlim(xLimits);
            ylim(yLimits);
            view(0,90);
            caxis([0 varargin{2}]);
        else
            axes('OuterPosition',[0 (1-2*imageSpace)/2 1/2 (1-2*imageSpace)/2]);
            surf(1:length(Img),waveScales,wavData1(:,:,end+rowShift),'EdgeColor','none','FaceColor','interp')
            xlim(xLimits);
            ylim(yLimits);
            view(0,90);
            caxis([0 cmax1]);
            title(['Wavelet transform data for ' num2str(-rowShift) ' rows above the cone'])
            axes('OuterPosition',[1/2 (1-2*imageSpace)/2 1/2 (1-2*imageSpace)/2]);
            surf(1:length(Img),waveScales,varargin{1},'EdgeColor','none','FaceColor','interp');
            xlim(xLimits);
            ylim(yLimits);
            view(0,90);
            caxis([0 varargin{2}]);
            for i = 3:2:length(varargin)-mod(length(varargin),2)
                axes('OuterPosition',[(i-3)/4 0 1/2 (1-2*imageSpace)/2]);
                surf(1:length(Img),waveScales,varargin{i},'EdgeColor','none','FaceColor','interp');
                xlim(xLimits);
                ylim(yLimits);
                view(0,90);
                caxis([0 varargin{i+1}]);
            end
        end
    else
        if numExtraData == 2
            axes('OuterPosition',[0 (1-2*imageSpace)/2 1 (1-2*imageSpace)/2]);
            surf(1:length(Img),waveScales,wavData1(:,:,end+rowShift),'EdgeColor','none','FaceColor','interp')
            xlim(xLimits);
            ylim(yLimits);
            view(0,90);
            caxis([0 cmax1]);
            title(['Wavelet transform data for ' num2str(-rowShift) ' rows above the cone'])
            for i = 1:2:length(varargin)-mod(length(varargin),2)
                axes('OuterPosition',[(i-1)/4 0 1/2 (1-2*imageSpace)/2])
                surf(1:length(Img),waveScales,varargin{i},'EdgeColor','none','FaceColor','interp')
                xlim(xLimits);
                ylim(yLimits);
                view(0,90);
                caxis([0 varargin{i+1}]);
            end
        else
            axes('OuterPosition',[0 2*(1-2*imageSpace)/3 1 (1-2*imageSpace)/3]);
            surf(1:length(Img),waveScales,wavData1(:,:,end+rowShift),'EdgeColor','none','FaceColor','interp')
            xlim(xLimits);
            ylim(yLimits);
            view(0,90);
            caxis([0 cmax1]);
            title(['Wavelet transform data for ' num2str(-rowShift) ' rows above the cone'])
            for i = 1:2:length(varargin)-mod(length(varargin),2)
                axes('OuterPosition',[(mod(i,4)-1)/4 (i<5)*(1-2*imageSpace)/3 1/2 (1-2*imageSpace)/3]);
                surf(1:length(Img),waveScales,varargin{i},'EdgeColor','none','FaceColor','interp')
                xlim(xLimits);
                ylim(yLimits);
                view(0,90);
                caxis([0 varargin{i+1}]);
            end
        end
    end
else
    axes('OuterPosition',[0 0 1 1-2*imageSpace]);
    surf(1:length(Img),waveScales,squeeze(wavData1(:,:,end+rowShift)),'EdgeColor','none','FaceColor','interp')
    caxis([0 cmax1]);
    title(['Wavelet transform data for ' num2str(-rowShift) ' rows above the cone'])
    xlim(xLimits);
    ylim(yLimits);
    view(0,90);
    axes('OuterPosition',[0 1-imageSpace 1 imageSpace]);
    imshow(image1./255)
    title(['Image Number ' num2str(img_no)]);
    axes('OuterPosition',[0 (1-2*imageSpace) 1 imageSpace]);
    image1 = zeros([size(image_ft),3]);
    image1(:,:,1) = scaleTo255(image_ft);
    image1(:,:,2) = scaleTo255(image_ft);
    image1(:,:,3) = scaleTo255(image_ft);
    imshow(image1./255)
end



end