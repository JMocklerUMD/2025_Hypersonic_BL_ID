function [secondMode,packets] = filteredImageIdentification_v2(shot, loopRange,...
    no_ref, furthest_image, forDisplay, turbulenceInfo, turbulent1)
% [secondMode,[turbulent_imgs]] = filteredImageIdentification(shot, loopRange, no_ref, furthestImage, forDisplay, turbulenceInfo,turbulent1)
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
%       turbulenceInfo - this vector contains the maximum FT response in
%           the range of the first harmonic for each image in the sequence.
%           This information is used in detecting turbulent spots
%       turbulent1 - the images which have been indicated to have
%           significant turbulent spots
%
%   Outputs:
%       secondMode - the images which had identified 2nd mode wave packets
%           in them
%   Optional Output:
%       turbulent_imgs - the images in which a turbulent spot has been
%           identified
%
%   This function identifies images with second mode wave packets as well
%   as images that are thought to have turbulent spots and possible
%   locations for those spots. The turbulence detection does not work well
%   with images that have significant pockets of highly varying density in
%   them. It does all of its identification using information from the
%   frequency domain and a filtered version of the image to determine if
%   there are any 2nd mode wave packets.
%

% setup information
% the global variables only really apply if the function is being used to
% display otherwise their status isn't important
global loopCount rowShift notDone
addpath([pwd '/wave_packet_detection'])
base_dir = pwd;


% get the correct function to read the images with
imgReadFxn = findImageReadFxn(shot);

% logic for display purposes.
notDone = 1==1;
% saved information for looking at later
% savedInfo = zeros(length(loopRange),3);

% number of points for the fourier transform
pts = 2^16;



img_num = loopRange;

turbulentImage = (sum(turbulent1 == img_num) == 1);





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


fprintf('%d \n', img_num);

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
    badBL = 1;
    error('Boundary layer is too thin. Change detection method or refine code.');
else
    badBL = 0;
end


% [~,~,~,dx] = imgReadFxn(base_dir,shot,img_num);


curr_img_filt = curr_img_filt*(2^bit_depth-1);


% initialize some values
secondMode = zeros(size(loopRange));
turbulent_imgs = secondMode;

% figure handles
if forDisplay
    figTitle = sprintf('2nd Mode Wave Packet Identification Shot %4i',shot);
    if forDisplay == 1 || forDisplay == 3
        h1 = figure('KeyPressFcn',@figureFunction,'Name',figTitle);
    else
        h1 = [];
    end
    if forDisplay == 2 || forDisplay == 3
        h2 = figure('KeyPressFcn',@figureFunction,'Name','Turbulence Information');
    else
        h2 =[];
    end
end
% loop stuff
loopCount = 1;
rowShift = 0;




% Fourier Transform
% run the fourier transform across each row
FT = fft(curr_img_filt(BL_height:cone,:),pts,2);
% get the power spectrum by getting |FT|^2
FT = FT.*conj(FT)/pts;
% the frequencies of interest
f = (0:pts/2)/pts;


% run a band pass filter over the boundary layer
bp_filter = fdesign.bandpass(1/(4.5*BL_thickness),(1.5)/(4.5*BL_thickness),...
    2/(1*BL_thickness),(3.2)/(1.5*BL_thickness),5,2,5);
bp_filter = design(bp_filter,'cheby1');
filtered_img = zeros(cone - BL_height+1,img_length);

% run a band pass filter over the boundary layer for 1st harmonic
bp_filter1 = fdesign.bandpass(2/(1.5*BL_thickness),(2.2)/(1.5*BL_thickness),...
    2/BL_thickness,2.2/BL_thickness,5,2,5);
bp_filter1 = design(bp_filter1,'cheby1');
% filter for finding turbulence (basically high pass but didn't want
% too much extra noise)
bp_filter2 = fdesign.bandpass(1.4/BL_thickness,1.5/BL_thickness,...
    4/BL_thickness,4.1/BL_thickness,5,2,5);
bp_filter2 = design(bp_filter2,'cheby1');
filtered_img1 = zeros(cone - BL_height+1,img_length);
filtered_img2 = zeros(cone,img_length);
% run the filters over the image to get the 2nd mode and 1st harmonic
% signals
for i = 1:cone
    if i >= BL_height
        filtered_img(i-BL_height+1,:) =  filter(bp_filter,curr_img_filt(i,:));
        if ~rotate % if you actually want to check to make sure you don't pick up turbulence
            % this check is only useful if the image has a sufficient
            % dynamic range. In the case of the images that needed
            % rotation in testing the range was too small.
            filtered_img1(i-BL_height+1,:) = filter(bp_filter1,curr_img_filt(i,:));
        end
    end
    filtered_img2(i,:) = filter(bp_filter2,curr_img_filt(i,:));
end




% Frequency bounds for 2nd mode frequency
find_min = find(f>1/(4.5*BL_thickness),1,'first');
find_max = find(f<(3/(BL_thickness*1.5)),1,'last');

% Check to see if there are any 2nd mode wave packets
% clear some variables
wave_packets = [];
starts = [];
stops = [];


% check to see if there is a significant peak in the frequency spectrum
% in the range of frequencies where we expect the 2nd mode to appear
if max(max(FT(:,find_min:find_max)))>5*mean(mean(FT)) && ~badBL
    
    % if there's a peak then look at the filtered image to determine if we
    % can find where the packet is by looking at the rows individually
    wave_packets = findWavePackets(filtered_img,filtered_img1,BL_thickness,...
        curr_img_filt(BL_height:cone,:),turbulentImage, 1);
    
    if sum(wave_packets(:,1,1))~=0
        
        % if we might have packets determine if they are really there by
        % comparing the information from all of the rows
        weighting = max((FT(:,find_min:find_max)),[],2)/...
            max(max(FT(:,find_min:find_max)));
        [starts,stops] = findBounds(wave_packets,weighting,BL_thickness);
        
        if ~isempty(starts) && ~isempty(stops)
            % pair off the boundaries so that there is the same number of
            % starts and stops
            Pairs = pairBounds(starts,stops,wave_packets,BL_thickness,weighting);
            starts = starts(Pairs(:,1));
            stops = stops(Pairs(:,2));
            % Save which images have 2nd mode wave packets
            if ~isempty(starts) && ~isempty(stops)
                secondMode = 1;
            end
        end
    end
end


% find turbulent spots
temp = find(f>4/BL_thickness,1,'first');
if isempty(temp)
    temp = length(f);
end
[spots, turbulentMeasure] = detectTurbulence(filtered_img2,FT,turbulenceInfo,...
    find(f<1/(BL_thickness),1,'last'),temp);

if spots(1,1) ~= 0
    turbulent_imgs(loopCount) = 1;
end

dx = 1;
% Display the image and the information used to identify wave packets
if forDisplay
    if forDisplay == 3
        wave_packets = [];
    end
    % display the image with frequency information and filtering
    % information
    
    if ~isempty(h1)
        displayPackets(curr_img,img_num,dx,wave_packets,cone,BL_height,starts,...
            stops,f,FT,filtered_img,filtered_img1,curr_img_filt,h1);
    end
    if ~isempty(h2)
        if isempty(spots)
            displayTurbulence(img_num, curr_img, curr_img_filt,turbulentMeasure, [],[], h2);
        else
            displayTurbulence(img_num, curr_img, curr_img_filt,turbulentMeasure, spots(:,1),spots(:,2), h2);
        end
    end
    
    % loop stuff
    pause()
    
end

if isempty(starts) && isempty(stops)
    packets = [];
else
    packets = [min(starts), max(stops), ones(numel(starts,1))*img_num];
end
% packets = [starts, stops, ones(numel(starts,1))*img_num];
% packets = floor(packets);

% if numel(packets) == 1
%     quick_view(curr_img_filt);
%     return;
% end

num_turb = 0;
turb_packets = zeros(0,3);

num_packets = size(packets,1);
colors = cell(num_packets + num_turb,1);
colors(1:num_packets) = {'second'};
colors(num_packets+1:end) = {'turb'};

% fig = figure(2);
% quick_view(curr_img_filt, [packets(:,1); turb_packets(:,1)],...
%     [packets(:,2); turb_packets(:,2)], fig, 0, colors);
% header = sprintf('Wavelet Decomposition: %d wave packet(s) (Shot %s, Image %d)',...
%     size(packets,1), num2str(shot), img_num);
% title(header);
% set(fig, 'position', [100 500 1800 350]);



% packets = cell(3,1);
% secondMode(2) = 0;
% secondMode(3) = 0;

end