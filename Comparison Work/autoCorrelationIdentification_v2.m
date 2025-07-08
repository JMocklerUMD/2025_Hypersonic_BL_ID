function [secondMode, packets] = autoCorrelationIdentification_v2(shot, loopRange,...
    no_ref, furthest_image, forDisplay,allStarts,allStops)
% [Image_nos] = autoCorrelationIdentification(shot, loopRange, ...
%           no_ref, furthestImage, forDisplay, allStarts, allStops)
%
%   Inputs:
%       shot - the number of the shot that you are interested in working
%           with.
%       loopRange - the images that you want to go through. If you are
%           looking to analyze a sequence of images pass in the entire
%           sequence.
%       no_ref - the number of reference images, from each side, to be
%           used to construct the average reference image
%       furthest_image - the last avaliable image in the sequence, this is
%           used to ensure that the averaged image doesn't attempt to put
%           in images that don't exist
%       forDisplay - a logical variable that indicates whether the images
%           should be displayed. If they are the program is paused after
%           each image is displayed.
%       allStarts - The start locations for each identified wave packet in
%           each image.
%       allStops - The stop locations for each identified wave packet in
%           each image.
%
%   Outputs:
%       Image_nos - the images which had identified 2nd mode wave packets
%           in them
%       allStarts - The start locations for each identified wave packet in
%           each image.
%       allStops - The stop locations for each identified wave packet in
%           each image.
%
%   This function identifies images with second mode wave packets by
%   pulling out chuncks of the image and seeing if they correlate with
%   other parts of the image in a patern that would suggest a second mode
%   wave packet.
%

% setup information
% the global variables only really apply if the function is being used to
% display otherwise their status isn't important
global loopCount rowShift notDone
addpath([pwd '/wave_packet_detection'])
base_dir = pwd;
% logic for display purposes.
notDone = 1==1;



img_num = loopRange;







%%%%%%%%
%Image processing logic for consistency
%%%%%%%%

close all;



if isnumeric(shot)
    shot = int2str(shot);
end

%%%%%%%%%%%%%%%%%%%%%%%
%read the shot parameter file
%%%%%%%%%%%%%%%%%%%%%%%


[shot, start, stop, re_ft, res, vert_reflect, horz_reflect,...
    reference_sub, gamma_adjust, pre_rot_row_crop, pre_rot_col_crop, ...
    rotate, fixed_rotation, post_rot_row_crop, post_rot_col_crop, ...
    auto_cone_crop, auto_rot_crop, auto_shadow_crop, bl_method, bl_adjust, bl_invert, peak_range,...
    power_thresh, turb_method, turb_thresh, img_dir, file_format, file_name] = read_par(shot);





%%%%%%%%%%%%%%%%%%%
% standard  setup information
%%%%%%%%%%%%%%%%%%%
addpath([pwd '/wave_packet_detection'])
base_dir = pwd;

% get the correct function to read the images with
imgReadFxn = findImageReadFxn(shot);


% first, get a reference image to build off of (and save some computations)
% this is the averaged image for 2*no_ref+1 images centered on 'img_no'
if regexp(sprintf('%s',shot), 'dmd')
    ref_img = imgReadFxn(base_dir,shot,0);
    ref_img = double(ref_img);
else
    [ref_img,nnPrev] = referenceImage(no_ref,stop,img_num,base_dir,shot,...
        0, 0, 0);
end
   

curr_img = imgReadFxn(base_dir,shot,img_num);
curr_img = double(curr_img);



%divide images by the maximum pixel values in order to standardize the
%PSD values encountererd when performing spectral analysis
info = imfinfo([img_dir '/' eval(file_name)]);
bit_depth = info.BitDepth;
ref_img = ref_img/(2^bit_depth-1);
curr_img = curr_img/(2^bit_depth-1);



%subtract the reference image weighting according to the parfile
curr_img_filt = (curr_img-reference_sub*ref_img);


%pre rotation cropping if specified in the parfile
if pre_rot_row_crop
    ref_img = ref_img(eval(pre_rot_row_crop),:);   
    curr_img = curr_img(eval(pre_rot_row_crop),:);   
    curr_img_filt = curr_img_filt(eval(pre_rot_row_crop),:);
end
if pre_rot_col_crop
    ref_img = ref_img(:,eval(pre_rot_col_crop));   
    curr_img = curr_img(:,eval(pre_rot_col_crop));   
    curr_img_filt = curr_img_filt(:,eval(pre_rot_col_crop));
end


%apply pixel reflection if necessary
%NOTE: this occurs after the first round of cropping
if vert_reflect
    ref_img = flip(ref_img, 1);  
    curr_img = flip(curr_img, 1);  
    curr_img_filt = flip(curr_img_filt, 1);
end
if horz_reflect
    ref_img = flip(ref_img, 2);  
    curr_img = flip(curr_img, 2);  
    curr_img_filt = flip(curr_img_filt, 2);
end






% now if you need to rotate the images do so [taken from Nathan's code]
if rotate
    if fixed_rotation  %if we have a constant, predetermined rotation angle
        ref_img = imrotate(ref_img,fixed_rotation,'bilinear');
        curr_img = imrotate(curr_img,fixed_rotation,'bilinear');
        curr_img_filt = imrotate(curr_img_filt,fixed_rotation,'bilinear');
        ang = fixed_rotation*pi/180;
    else
        %adjust the reference image, will have better cone definition
        %rowInds and colInds are new indices in case dimensionality is reduced
        [ref_img,yp,rowInds,colInds] = coneAdjuster(clamp(ref_img));
        temp = fit((1:length(yp))',yp','poly1');
        ang = atan(temp.p1);
        
        %rotate the current unsubtracted image
        curr_img = imrotate(curr_img,ang*180/pi,'bilinear');
        curr_img = curr_img(rowInds,colInds);
        
        %rotate the current referenced image
        curr_img_filt = imrotate(curr_img_filt,ang*180/pi,'bilinear');
        curr_img_filt = curr_img_filt(rowInds,colInds);
    end 
end


%crop the end of the cone if it appears in the current frame
if auto_cone_crop
    %get the vertical edge magnitude and find the column at which it drops
    %off below the average cone magnitude jump
    ref_img = clamp(ref_img);
    Edge_vert = 1/8*(-ref_img(1:end-2,1:end-2)+ref_img(3:end,1:end-2)-2*ref_img(1:end-2,2:end-1)+2*ref_img(3:end,2:end-1)-ref_img(1:end-2,3:end)+ref_img(3:end,3:end));
    cone_edge_mag = mean(mean(abs(Edge_vert(end-5:end,:))));
    end_cone_mag = mean(abs(Edge_vert(end-5:end,ceil(size(ref_img,2)/2):end)));
    end_cone_col = find(end_cone_mag < cone_edge_mag, 1, 'first' ) + ceil(size(ref_img,2)/2);
    ref_img = ref_img(:, 1:end_cone_col);
    curr_img = curr_img(:, 1:end_cone_col);
    curr_img_filt = curr_img_filt(:, 1:end_cone_col);
end



%crop the blank space above the cone caused by rotation
if auto_rot_crop
    trunc_row = ceil(size(ref_img, 2)*tan(abs(ang)));
    ref_img = ref_img(trunc_row:end, :);
    curr_img = curr_img(trunc_row:end, :);
    curr_img_filt = curr_img_filt(trunc_row:end, :);
end



%post rotation cropping if specified in the parfile
if post_rot_row_crop
    fprintf('%d -- %s -- [%d %d]\n', img_num, post_rot_row_crop, size(ref_img, 1), size(ref_img, 2));
    ref_img = ref_img(eval(post_rot_row_crop),:);
    curr_img = curr_img(eval(post_rot_row_crop),:);
    curr_img_filt = curr_img_filt(eval(post_rot_row_crop),:);
end
if post_rot_col_crop
    ref_img = ref_img(:,eval(post_rot_col_crop));
    curr_img = curr_img(:,eval(post_rot_col_crop));
    curr_img_filt = curr_img_filt(:,eval(post_rot_col_crop));
end




%run boundary layer finding algorithm
if bl_adjust
    temp_img_str = sprintf('imadjust(clamp(ref_img), %s)', bl_adjust);
    temp_img = eval(temp_img_str);
    if bl_invert  %invert the image for bl finding isf necessary
        temp_img = imcomplement(temp_img);
    end
    [cone,BL_height,BL_thickness] = findBL(temp_img, bl_method);
else   %if no gamma adjustments needed
    if bl_invert
        temp_img = imcomplement(clamp(ref_img));
    else
        temp_img = clamp(ref_img);
    end
    [cone,BL_height,BL_thickness] = findBL(temp_img, bl_method);
end





%4020 has a poorly defined boundary layer; correct for that here
if shot==4020 && BL_thickness > 15
    BL_height = BL_height + BL_thickness - 8;
    BL_thickness = 8;
end






%if we want to crop a schlieren shadow that precedes the relevent field of view
if auto_shadow_crop
    crop_col = crop_shadow(curr_img_filt);
    ref_img = ref_img(:,crop_col:end);
    curr_img = curr_img(:,crop_col:end);
    curr_img_filt = curr_img_filt(:,crop_col:end);
end



%we are done with most of the main image processing; just map to [0,1] now
%curr_img_filt = clamp(curr_img_filt);

%if there are variations in background intensity, take a projection across
%the rotated reference image and divide the currend image by this
nonuniform_background = 0;
if nonuniform_background
    intensity_curve = smooth(mean(ref_img(cone-7:cone-2,:),1),51);
    intensity_curve = intensity_curve/max(intensity_curve);
    intensity_corr_img = ones(size(curr_img_filt,1),1)*intensity_curve';
    curr_img_filt = curr_img_filt./intensity_corr_img;
end


%adjust the gamma correction if needed
if gamma_adjust
    curr_img_filt = imadjust(curr_img_filt, [], [], gamma_adjust);
end

img_length = length(curr_img_filt);




%before we proceed, is the bl thick enough
if BL_thickness < 3
    error('Boundary layer is too thin. Change detection method or refine code.');
end














% number of points for the fourier transform
pts = 2^16;
% [~,~,~,dx] = imgReadFxn(base_dir,shot,img_num);

% figure handles
if forDisplay
    figTitle = sprintf('2nd Mode Wave Packet Identification Shot %4i',shot);
    h1 = figure('KeyPressFcn',@figureFunction,'Name',figTitle);
    %     h2 = figure('KeyPressFcn',@figureFunction,'Name','Turbulence Information');
end

% loop stuff
loopCount = 1;
rowShift = 0;
pause on % this is for display purposes


% run the fourier transform across each row
FT = fft(curr_img_filt(BL_height:cone,:),pts,2);
% get the power spectrum by getting |FT|^2
FT = FT.*conj(FT)/pts;
% the frequencies of interest
f = (0:pts/2)/pts;

% run a band pass filter over the boundary layer
bp_filter = fdesign.bandpass(1.25/(4.5*BL_thickness),(1.5)/(4.5*BL_thickness),...
    2/(1*BL_thickness),(3.2)/(1.5*BL_thickness),10,2,10);
bp_filter = design(bp_filter,'cheby1');
filtered_img = zeros(cone - BL_height,img_length);

% run a band pass filter over the boundary layer for 1st harmonic
bp_filter1 = fdesign.bandpass(2/(1.5*BL_thickness),(2.2)/(1.5*BL_thickness),...
    2/BL_thickness,2.2/BL_thickness,10,2,10);
bp_filter1 = design(bp_filter1,'cheby1');
% filter for finding turbulence (basically high pass but didn't want
% too much extra noise)
bp_filter2 = fdesign.bandpass(1.4/BL_thickness,1.5/BL_thickness,...
    4/BL_thickness,4.1/BL_thickness,10,2,10);
bp_filter2 = design(bp_filter2,'cheby1');
filtered_img1 = zeros(cone - BL_height+1,img_length);
filtered_img2 = zeros(cone,img_length);
% run the filters over the image to get the 2nd mode and 1st harmonic
% signals
for i = 1:cone
    if i >= BL_height
        filtered_img(i-BL_height+1,:) =  filter(bp_filter,curr_img_filt(i,:));
        filtered_img1(i-BL_height+1,:) = filter(bp_filter1,curr_img_filt(i,:));
    end
    filtered_img2(i,:) = filter(bp_filter2,curr_img_filt(i,:));
end


% Soft Thresholding (sign(x)*(abs(x)-mean)*(abs(x)>mean) is used for images
% with sufficient dynamic range
image_ft1 = sign(curr_img_filt).*(abs(curr_img_filt)-mean2(curr_img_filt)).*...
    ((curr_img_filt)>(mean2(curr_img_filt)));

% Check to see if there are any 2nd mode wave packets
% clear some variables
wave_packets = [];

% use a window of 2*delta as used by Casper et al.
windowSize = 2*(BL_thickness+1);
% run the window over all but the last bit of the image
steps = 0:length(curr_img_filt)-2*windowSize;
correlationResults = zeros(length(steps),length(curr_img_filt));
packetEnds = zeros(size(steps));
for i = steps
    start = i+1;
    % get the section to do the fourier transorm on and apply the
    % window
    section = image_ft1(BL_height:cone,start:start+windowSize);
    for col = start:length(curr_img_filt)-windowSize-1
        correlationResults(i+1,col) = corr2(section,...
            image_ft1(BL_height:cone,col:col+windowSize));
    end
    % determine if there is a wavepacket
    data = smoothdata(smoothdata(correlationResults(i+1,start:end)));
    [peaks,locs] = findpeaks(data); % info to be sued
    isPacket = 1; % logic
    peakInd = 1; % index variable
    sepMax = 3.5*BL_thickness; % separation limits
    sepMin = 1.8*BL_thickness;
    numPeaks = 1;
    % the loop (if it runs forever it's probably stuck here)
    while isPacket
        nextPeak = peakInd+1;
        if nextPeak <=length(locs) % make sure you're not at the end
            while peaks(nextPeak) < 0 % if the peak isn't any good try the next one
                if nextPeak == length(locs)
                    break
                else
                    nextPeak = nextPeak+1;
                end
            end
            distance = locs(nextPeak)-locs(peakInd);
            if  distance > sepMax || distance < sepMin % check if it fits
                isPacket = 0;
            else
                numPeaks = numPeaks+1;
                peakInd = nextPeak;
            end
        else
            isPacket = 0;
        end
    end
    if numPeaks > 3-2*(rotate) || (length(data)<2*windowSize && peakInd ~= 1)
        packetEnds(i+1) = locs(peakInd)+i; % save the location if it's worth saving
    end
    
end

% determine the "true" ends

% variables that we'll need
notTaken = packetEnds~=0;
eSupporters = [];
newSupporters = [];
stops = [];
starts = [];
tempInd = 1;
while sum(notTaken) ~=0; % while there are still ends that you haven't looked at yet
    if isempty(eSupporters) % starting over
        temp = packetEnds(notTaken==1);
        eMean = temp(find(abs(temp-median(temp))==min(abs(temp-median(temp))),1));
    else % adjusting if it's changed
        eMean = mean(packetEnds(eSupporters));
    end
    if eMean-30<0 % make sure you don't include 0 in your check
        minVal = 0;
    else
        minVal = (eMean-30);
    end
    newSupporters = find(((packetEnds.*notTaken)>minVal).*...
        ((packetEnds.*notTaken)<(eMean+30))); % find who's around your new mean
    if isempty(setxor(newSupporters,eSupporters))
        % done
%         if length(eSupporters) > 40
        if length(eSupporters) > BL_thickness*4
            % save if it has enough supporters
            stops(tempInd) = eMean+windowSize;
            starts(tempInd) = min(eSupporters);
            tempInd = tempInd+1;
        end
        % reset
        notTaken(eSupporters) = 0;
        eSupporters = [];
        newSupporters = [];
    else
        % not done
        eSupporters = newSupporters;
    end
    
end

% check for overlap

tempStarts = []; % temp variables so we don't loose any important data
tempStops = [];
tempInd = 1;
for i = 1:length(starts)
    tempLength = stops(i)-starts(i); % length of the packet
    overlapL = (starts(i)<stops).*(stops(i)>stops).*(stops~=0);
    overlapR = (stops(i)>starts).*(starts(i)<starts).*(starts~=0);
    if sum(overlapL.*overlapR) > 0
        starts(i) = 0;
        stops(i) = 0;
    elseif sum(overlapL)>0
        % if there is overlap on the left side of the packet combine them
        % if the overlap is sufficiently large
        tempInds = find(overlapL);
        for j = tempInds
            if stops(j)-starts(i)>0.25*tempLength && starts(i) ~= 0
                tempStops(tempInd) = stops(i);
                tempStarts(tempInd) = starts(j);
                starts(i) = 0;
                starts(j) = 0;
                stops(i) = 0;
                stops(j) = 0;
                tempInd = tempInd+1;
            end
        end
    elseif sum(overlapR) >0
        % if there is overlap on the right side of the packet combine them
        % if the overlap is sufficiently large
        tempInds = find(overlapR);
        for j = tempInds
            if stops(i)-starts(j)>0.25*tempLength && starts(j) ~= 0
                tempStops(tempInd) = stops(j);
                tempStarts(tempInd) = starts(i);
                starts(i) = 0;
                starts(j) = 0;
                stops(i) = 0;
                stops(j) = 0;
                tempInd = tempInd+1;
            end
        end
    else
        % if there isn't overlap then just continue
        tempStops(tempInd) = stops(i);
        tempStarts(tempInd) = starts(i);
        tempInd = tempInd+1;
    end
    
end
starts = tempStarts(tempStarts~=0);
stops = tempStops(tempStops~=0);

% save in variables to be returned
allStarts(loopCount,1:length(starts)) = starts;
allStops(loopCount,1:length(stops)) = stops;


if ~isempty(starts)
    secondMode(loopCount) = 1;
else
    secondMode = 0;
end

dx = 1;

% Display the image and the correlation information
if forDisplay
    % display the image with frequency information and filtering
    % information
    displayPackets(curr_img_filt,img_num,dx,wave_packets,cone,BL_height,starts,...
        stops,f,FT,filtered_img,filtered_img1,curr_img_filt,h1);
    % loop stuff
    
end



%set packets variable for parallel operation
packets = cell(3, 1);
packets{1} = [allStarts', allStops'];
%secondMode(2:3) = 0;



end