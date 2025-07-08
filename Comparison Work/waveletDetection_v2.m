function [second_mode, wave_packets] = waveletDetection_v2(shot,img_num,no_ref,...
    furthest_image,upperFactor,forDisplay)
% [secondMode] = waveletDetection(base_dir,shot,loopRange,no_ref,furthest_image,upperFactor,forDisplay)
%
%   Inputs:
%       base_dir - the directory where the main functions are located
%       shot - The shot number for this image sequence
%       loopRange - The images to be analyzed
%       no_ref - the number of images on each side to be used in the
%           reference image
%       furthest_image - The last image in the sequence
%       upperFactor - a factor used in thresholding (it modifies
%           dropOffPercent and is usually >=1)
%       forDisplay - Indicates whether to display the results or not
%
%   Output:
%       secondMode - A vector of the same size as loopRange which indicates
%           whether that image in loopRange has a second-mode wave packet
%           or not
%
% This function uses the wavelet analysis technique to determine whether an
% image has a second mode wave packet or not and where it is located. The
% wavelet analysis is done by treating each row as an individual signal and
% doing a continous wavelet transform on each row to determine the location
% wave packet in that row. Then the results from the individual rows are
% compiled to give the final result.
%


global loopCount notDone rowShift moveEvent
addpath([pwd '/wave_matlab/']);
addpath([pwd '/wave_packet_detection/']);



% factors for wavelet analysis
bandCenter = 3; % center of scale band to look for wave packets (bandCenter*BL_thickness)
narrowFactor = 1.5; % width/2 of band for looking for wave packets (narrowFactor*BL_thickness)
wideFactor = 2.5;
peakCheckPercent = 0.8; % ensure that the peak in the wavelet transform is wide enough
dropOffPercent = 0.5; % percentage of peak's value that is considered end of wave packet in transformed space
scaleStep = 0.05; % the step size between scales (at least to start)
packetLengthFactor = 9; % minimum number of BL_thicknesses needed to be considered a wave packet
overlapFactor = 3; % variation in bounds for checking overlap (overlapFactor*BL_thickness)

vCheckSpacing = 1; % factor used in Method 2 for determining the spacing to check for scale extent
vCheckFactor = 0.8;
% vCheckNumber = 13; % half the verticle distribution allowed

% factors used in Method 1 for checking the extent in the scale direction
lowerFactor = 1/2;
% upperFactor = upperFactor;
lowerPercent = 0.2;
upperPercent = 0.2;

% function for reading images
%imgReadFxn = findImageReadFxn(shot);
% if displaying make sure you have a window





fprintf('\n%d', img_num);



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
%  curr_img_filt = clamp(curr_img_filt);

%adjust the gamma correction if needed
if gamma_adjust
    curr_img_filt = imadjust(curr_img_filt, [], [], gamma_adjust);
end

img_length = length(curr_img_filt);




%before we proceed, is the bl thick enough
if BL_thickness < 3
    error('Boundary layer is too thin. Change detection method or refine code.');
end
















% actual wavelet analysis begins here
narrowBand  = [(bandCenter-narrowFactor)*BL_thickness (bandCenter+narrowFactor)*BL_thickness];
wideBand = [(bandCenter-wideFactor)*BL_thickness (bandCenter+wideFactor)*BL_thickness];
wave_packets = zeros(BL_thickness+1,1,2);
weighting = zeros(BL_thickness+1,1);

%testing for revised wavelet identification
%wavelet_hist = zeros(190, size(curr_img_filt,2), numel(BL_height:cone));
iii = 1;

scaleStep = 0.01;

for i = BL_height:cone
    % using the wavelet transform from Torrence and Compo to get the
    % wavelet transform with the Morlet wavelet. (this will need to be
    % modified if you are using a different software package to
    % calculate the wavelet transform)
    [wTransform,period] = wavelet(smoothdata(curr_img_filt(i,:)),1,1,scaleStep);
    
    wavelet_hist(:,:,iii) = abs(wTransform).^2;
    iii = iii + 1;
end


secondmode_band = BL_thickness*[2,3.5];
turb_band = BL_thickness*[0.5, 1.6];
turb_band = BL_thickness*[0.2857, 0.6667];
wavelet_avg = mean(wavelet_hist, 3);

if shot == 1297 || shot == 1296
    smoothing = 31;
else 
    smoothing = 3;
end

secondmode_ind = find(period>=secondmode_band(1),1,'first'):find(period<=secondmode_band(2),1,'last');
turb_ind = find(period>=turb_band(1),1,'first'):find(period<=turb_band(2),1,'last');
turb_mean = mean(wavelet_avg(turb_ind, :),1);
turb_mean = smoothdata(turb_mean);

%calculate the ratio of secondmode to turbulent signals and smooth
secondmode_ratio = mean(wavelet_avg(secondmode_ind, :),1)./mean(wavelet_avg(turb_ind, :),1);
secondmode_ratio = smoothdata(secondmode_ratio);

secondmode_mean = smoothdata(mean(wavelet_avg(secondmode_ind, :),1));
secondmode_mean(end) = 0;

%get canny image for turbulence detection
canny_thresh = 0.4;
canny_edges = edge(curr_img_filt,'canny', canny_thresh);


%now get bounds of wavepackets
found = 0;
i = 1;
prev_col = 1;

%thresh = 1.0;   %OK FOR 1296 ONLY
thresh = 0.005;
if shot==4017; thresh = 0.0022; thresh = 0.0022*1.2;end
if shot==4016; thresh = 1.275*6.4608e-04; end  %4016
if shot==1297; thresh = 1.1*33.2529; thresh = 1.725*33.2529; thresh = 1.05*50.2158;end
if shot==1296; thresh = 50.2158;     thresh = 1.35*50.2158; end
if shot==4119; thresh = 0.022;  thresh = 0.035;end   %higher threshold for strong packets
if shot==450;  thresh = 0.0035; end
if shot==451;  thresh = 0.0035; end
if shot==452;  thresh = 0.0035; end
if shot==453;  thresh = 0.0035; end
if shot==30;   thresh = 50; end
if shot==16;   thresh = 0.01; end
if shot==91;   thresh = 0.006; end
if shot==92;   thresh = 0.007; end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plotting things for the paper
% figure(3);
% wavelet_avg_tmp = clamp((wavelet_avg).^(1/3));
% wavelet_avg_tmp = imadjust(wavelet_avg_tmp, [0.0; 1.0], [], 1.20);
% % imagesc(1:size(wavelet_avg_tmp, 2), 1./(period/BL_thickness), wavelet_avg_tmp);
% surf(1:size(wavelet_avg_tmp, 2), 1./(period/BL_thickness), wavelet_avg_tmp, 'edgecolor', 'none')
% view([0 90])
% axis tight;
% ylim([min(BL_thickness./period), 3]);
% hold on;
% plot3([1,size(wavelet_avg_tmp,2)], BL_thickness*[1/secondmode_band(1), 1/secondmode_band(1)],...
%     [1e6 1e6], '--r', 'linewidth', 2);
% plot3([1,size(wavelet_avg_tmp,2)], BL_thickness*[1/secondmode_band(end), 1/secondmode_band(end)],...
%     [1e6 1e6], '--r', 'linewidth', 2);
% % daspect([1 4 1]);
% colormap gray;
% ylabel('Wavenumber [1/$\delta$]', 'interpreter','latex');
% xlabel('Length Along Image [Pixels]');
% set(gca, 'fontsize', 15);
% 
% figure(4);
% plot(secondmode_mean)
% hold on;
% plot([1, numel(secondmode_mean)], [thresh, thresh], '--k', 'linewidth', 2);
% xlabel('Length Along Image [Pixels]');
% ylabel('Power Spectral Density');
% set(gca, 'fontsize', 15);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



packets = zeros(20,3);
while ~found
    %get start of packet
    try
        packets(i, 1) = find(secondmode_mean(prev_col:end) > thresh, 1, 'first') + prev_col - 1;
%         packets(i, 1) = find(secondmode_ratio(prev_col:end) > thresh, 1, 'first') + prev_col - 1;
    catch
        found = 1;
        continue;
    end
    prev_col = packets(i, 1);
    
    %get end of packet
    try
        packets(i, 2) = find(secondmode_mean(prev_col:end) < thresh, 1, 'first') + prev_col - 1;
%         packets(i, 2) = find(secondmode_ratio(prev_col:end) < thresh, 1, 'first') + prev_col - 1;
    catch
        packets(i,:) = [];
         found = 1;
        continue;
    end
        prev_col = packets(i, 2);

    packets(i,3) = img_num;
    
    i = i + 1;
    
end



%if packets are only 3 wavelengths apart, combine them 
i = 1;
while i+1 <= size(packets,1)
    min_waves = 3;
    if (abs(packets(i+1, 1) - packets(i,2)) < (BL_thickness*2)*min_waves)
        packets(i,2) = packets(i+1,2);
        packets(i+1,:) = [];
    else
        i = i + 1;
    end
end
packets(end,:) = [];


%turbulent thresholds
if shot == 4017
    max_turb = 1.5e-4;
    max_turb = 0.7-4;
elseif shot == 4119
    max_turb = 2.2052e-04;
elseif shot == 1296
    max_turb = 0.930e-04;
    max_turb = 0.60e-04;
elseif shot == 4016
    max_turb = 1.3e-04;
elseif shot == 1297
    max_turb = 1.0052e-04;
elseif shot == 450
    max_turb = 1e-04;
elseif shot == 451
    max_turb = 1.2e-04;
elseif shot == 452
    max_turb = 1e-04;
elseif shot == 453
    max_turb = 1e-04;
elseif shot == 30
    max_turb = 2e-04;
elseif shot == 16
    max_turb = 1e-03;
elseif shot == 91
    max_turb = 2e-04;
elseif shot == 92
    max_turb = 2e-04;
else
    max_turb = 1.2052e-04;
end

%post-process packets by checking for size, etc.
i = 1;
while i <= size(packets,1)
    %size and power constraints on packets (3.5 waves)
    min_waves = 0.5;
    min_second = 1*thresh;  %should set to 1.1
        
    %check for wavepacket_width
    if packets(i, 2) - packets(i,1) < (BL_thickness*2)*min_waves 
        packets(i,:) = [];

    %place constraints on mean power in wavepacket
    elseif mean(secondmode_mean(packets(i,1):packets(i,2))) < min_second
%     elseif mean(secondmode_ratio(packets(i,1):packets(i,2))) < min_second
        packets(i,:) = [];
       
    %check for highF turbulence in each wavepacket first
    elseif mean(turb_mean(packets(i, 1):packets(i,2))) > max_turb
        
        %now check for turbulence using canny method
        section = canny_edges(1:BL_height-4, packets(i,1):packets(i,2));
        [~, edge_c] = find(section);
        edge_c = unique(edge_c);
        if numel(edge_c) < 0.5*size(section,2);
            i = i + 1;
        else
            packets(i,:) = [];
        end
       
    else
        i = i + 1;
    end
end




%find the beginning and end of turbulent spots in the current image
found = 0;
i = 1;
prev_col = 1;
turb_packets = zeros(20,3);
while ~found
    %get start of packet
    try
        turb_packets(i, 1) = find(turb_mean(prev_col:end) > max_turb, 1, 'first') + prev_col - 1;
        %packets(i, 1) = find(secondmode_ratio(prev_col:end) > thresh, 1, 'first') + prev_col - 1;
    catch
        found = 1;
        continue;
    end
    prev_col = turb_packets(i, 1);
    
    %get end of packet
    try
        turb_packets(i, 2) = find(turb_mean(prev_col:end) < max_turb, 1, 'first') + prev_col - 1;
        %packets(i, 2) = find(secondmode_ratio(prev_col:end) < thresh, 1, 'first') + prev_col - 1;
    catch
        turb_packets(i,:) = [];
        found = 1;
        continue;
    end
    prev_col = turb_packets(i, 2);
    turb_packets(i,3) = img_num;
    i = i + 1;
end
turb_packets(find(sum(turb_packets,2)==0), :) = [];

%merge turbulent spots if separated by less than 3 delta
num_turb = size(turb_packets, 1);
i = 1;
while i < num_turb
    if (abs(turb_packets(i,2) - turb_packets(i+1,1)) < 3*BL_thickness)
        turb_packets(i,2) = turb_packets(i+1,2);
        turb_packets(i+1,:) = [];
        num_turb = num_turb - 1;
    else
        i = i + 1;
    end
end
i = 1;
while i <= num_turb
    if (abs(turb_packets(i,2) - turb_packets(i,1)) < 2*BL_thickness)
        turb_packets(i, :) = []; 
        num_turb = num_turb - 1;
    else
        i = i + 1;
    end
end


%remove wave-packets that coincide with turbulence
train_num = 1;
train_size = size(packets,1);
i = 1;
while train_num <= size(packets,1)
    packet_cell = in_turbulence(packets(train_num,:), turb_packets);
    %[train{i}(train_num,:), train{3}] = in_turbulence(train{i}(train_num,:), train{3});
    if numel(packet_cell{1,1}) == 2
        packets(train_num,:) = packet_cell{1,1};
    elseif numel(packet_cell{1,1}) == 0
        packets(train_num,:) = [];
    else  %need to consider case in which wavepackets split into 3 parts
        packets(end+1,:) = [0,0,0];
        packets(train_num:end+size(packet_cell{i,1},1)-1,:) = [packet_cell{i,1};
            packets(train_num+1:end,:)];
        packets(end,:) = [];
    end
    
    if (packet_cell{2,1})
        turb_packets = packet_cell{2,1};
    else
        turb_packets = zeros(0,3);
    end
    
    %see if we reduce the number of packets
    if size(packets,1) < train_size
        train_size = size(packets,1);
    else
        train_num = train_num + 1;
    end
    
end

%check one more time that the wavepackets are wide enough
i = 1;
while i <= size(packets,1)
    %size and power constraints on packets
    min_waves = 3.5;
    %check for wavepacket_width
    if packets(i, 2) - packets(i,1) < (BL_thickness*2)*min_waves 
        packets(i,:) = [];
    else
        i = i + 1;
    end
end


if size(packets, 1) > 0
    second_mode(1) = 1;
else
    second_mode(1) = 0;
end
second_mode(2) = 0;
second_mode(3) = 0;

 
% %UNCOMMENT FOR TESTING
% num_packets = size(packets,1);
% colors = cell(num_packets + num_turb,1);
% colors(1:num_packets) = {'second'};
% colors(num_packets+1:end) = {'turb'};
% 
% fig = figure(2);
% quick_view(curr_img_filt, [packets(:,1); turb_packets(:,1)],...
%     [packets(:,2); turb_packets(:,2)], fig, 0, colors);
% header = sprintf('Wavelet Decomposition: %d wave packet(s) (Shot %s, Image %d)',...
%     size(packets,1), num2str(shot), img_num);
% title(header);
% set(fig, 'position', [100 500 1800 350]);




% if second_mode(1) == 0
%     packets = [];
% end
% if second_mode(3) == 0
%     turb_packets = [];
% end

wave_packets = cell(3,1);
wave_packets{1} = packets;
wave_packets{2} = [];
wave_packets{3} = turb_packets;

return; 


for i = 1:0

    % We only really care about the magnitude of the transform so save
    % that for later display
    if i == BL_height
        compiledTransform = abs(wTransform);
    else
        compiledTransform(:,:,i-BL_height+1) = abs(wTransform);
    end
    % get the indicies for the ranges of interest
    indsN = find(peroid>=narrowBand(1),1,'first'):find(peroid<=narrowBand(2),1,'last');
    indsW = find(peroid>=wideBand(1),1,'first'):find(peroid<=wideBand(2),1,'last');
    % isolate the section around the expected second mode wavelength
    sectionN = abs(wTransform(indsN,:));
    
    % save the max value in the narrow section for use in weighting
    weighting(i-BL_height+1) = max(max(sectionN));
    
    % set up to locate the peaks in each row. For each peak make sure
    % that it has a width > 4*delta by checking to see how quickly the
    % magnitude drops off and making sure that it is the maximum value
    % at that location
    
    minPeak = 1*mean2(sectionN);
    minThreshold = 0.75*minPeak;
    peakRow = [];
    peakCol = [];
    peakVal = [];
    tInd = 1;
    for j = 1:length(indsN)
        % get the local maximum
        [peak,loc] = findpeaks(sectionN(j,:));
        for k = 1:length(loc)
            % determine if they're worth keeping
            % by checking to make sure they don't drop off too fast in
            % the "x" direction
            if loc(k) <= 2*BL_thickness
                if sectionN(j,1)<peakCheckPercent*peak(k) && ...
                        sectionN(j,loc(k)+2*BL_thickness) > peakCheckPercent*peak(k) && ...
                        peak(k)>minPeak
                    % make sure that it is a maximum in the "y" direction
                    [~,locY] = findpeaks(sectionN(:,loc(k)));
                    if sum(locY==j) > 0
                        peakRow(tInd) = j;
                        peakCol(tInd) = loc(k);
                        peakVal(tInd) = peak(k);
                        tInd = tInd+1;
                    end
                end
            elseif loc(k)+2*BL_thickness>=length(IM)
                if sectionN(j,loc(k)-2*BL_thickness) > peakCheckPercent*peak(k) && ...
                        sectionN(j,end) > peakCheckPercent*peak(k) && ...
                        peak(k)>minPeak
                    % make sure that it is a maximum in the "y" direction
                    [~,locY] = findpeaks(sectionN(:,loc(k)));
                    if sum(locY==j) > 0
                        peakRow(tInd) = j;
                        peakCol(tInd) = loc(k);
                        peakVal(tInd) = peak(k);
                        tInd = tInd+1;
                    end
                end
            elseif (sectionN(j,loc(k)-2*BL_thickness) > peakCheckPercent*peak(k) || ...
                    sectionN(j,loc(k)+2*BL_thickness) > peakCheckPercent*peak(k)) && ...
                    peak(k)>minPeak
                % make sure that it is a maximum in the "y" direction
                [~,locY] = findpeaks(sectionN(:,loc(k)));
                if sum(locY==j) > 0
                    peakRow(tInd) = j;
                    peakCol(tInd) = loc(k);
                    peakVal(tInd) = peak(k);
                    tInd = tInd+1;
                end
            end
        end
    end
    
    tInd = 1;
    for j = 1:length(peakRow)
        % finding potential packet boundaries by looking at where this
        % row and the two adjcent rows drop off
        section1 = sectionN(peakRow(j)-1:peakRow(j)+1,peakCol(j):-1:1);
        section2 = sectionN(peakRow(j)-1:peakRow(j)+1,peakCol(j):end);
        section1 = section1<peakVal(j)*dropOffPercent;
        section2 = section2<peakVal(j)*dropOffPercent;
        section1 = sum(section1,1);
        section2 = sum(section2,1);
        leftBound = peakCol(j)+1-find(section1==3,1,'first');
        rightBound = peakCol(j)-1+find(section2==3,1,'first');
        if isempty(leftBound)
            leftBound = 1;
        end
        if isempty(rightBound)
            rightBound = length(IM);
        end
        % check to make sure that there isn't too much dispersion in
        % the direction of increasing and decreasing scale (wavelength)
        % to make sure that this isn't a turbulent spot.
        
        
        tIndicies = leftBound:floor(BL_thickness*vCheckSpacing):rightBound;
        if sum(tIndicies==rightBound) == 0
            % this is in case floor(BL_thickness*vCheckSpacing) doesn't
            % evenly divide rightBound-leftBound+1
            tIndicies = [tIndicies rightBound];
        end
        upperMaxes = zeros(1,length(tIndicies));
        lowerMaxes = upperMaxes;
        threshold = max(peakVal(j)*1*dropOffPercent,minThreshold);
        for k = tIndicies
            % grab verticle sections starting at the particular point
            % and going up or down
            section1 = abs(wTransform(indsN(peakRow(j)):end,k));
            section2 = abs(wTransform(indsN(peakRow(j)):-1:1,k));
            
            % find the first spot where the the wavlet transform value
            % drops below the threshold
            temp = find(section1<threshold,1,'first')-1;
            if isempty(temp)
                temp = length(section1);
            end
            % figure out what peroid difference that corrosponds to
            upperMaxes(tIndicies==k) = peroid(indsN(peakRow(j))+temp-1)-...
                peroid(indsN(peakRow(j)));
            % do the same thing on the other side
            temp = find(section2<threshold,1,'first')-1;
            if isempty(temp)
                temp = length(section2);
            end
            lowerMaxes(tIndicies==k) = peroid(indsN(peakRow(j)))-...
                peroid(indsN(peakRow(j))-temp+1);
        end
        vExtentCheck = sum((upperMaxes+lowerMaxes)>2*vCheckFactor*BL_thickness)==0;
        
        % additionally some diagonals should be checked to make sure
        % it's not distorted too much
        if vExtentCheck
            tcol0 = peakCol(j);
            trow0 = indsN(peakRow(j));
            trow = find(peroid(1:trow0)<(peroid(trow0)-vCheckFactor*BL_thickness),1,'last');
            angles = pi/16:(atan((trow0-trow)/(rightBound-...
                peakCol(j)))-pi/16)/8:atan((trow0-trow)/(rightBound-peakCol(j)));
            
            for angle = angles
                trows1 = ceil(tan(angle)*((1:length(IM))-tcol0)+trow0);
                trows2 = ceil(-tan(angle)*((1:length(IM))-tcol0)+trow0);
                section1 = zeros(size(wTransform));
                section2 = section1;
                % make matricies that have the line of interest as 1s
                firstCol1 = 1;
                firstCol2 = 1;
                lastCol1 = length(IM);
                lastCol2 = lastCol1;
                for k = 1:length(IM)
                    trow = ceil(tan(angle)*(k-tcol0)+trow0);
                    if trow >0 && trow <= size(section1,1)
                        section1(trow,k) = 1;
                    elseif trow <= 0
                        firstCol1 = k+1;
                    elseif trow > size(section1,1) && lastCol1 > k-1
                        lastCol1 = k-1;
                    end
                    trow = ceil(-tan(angle)*(k-tcol0)+trow0);
                    if trow >0 && trow <= size(section2,1)
                        section2(trow,k) = 1;
                    elseif trow > size(section2,1)
                        firstCol2 = k+1;
                    elseif  trow < 1 && lastCol2 > k-1
                        lastCol2 = k-1;
                    end
                end
                section1 = abs(wTransform(section1==1));
                section2 = abs(wTransform(section2==1));
                % now section1 is the line going up to the right and section2
                % is the line going down to the right. Check to make sure that
                % the spot still meets specifications in these directions
                tIndC = find(section1(tcol0-firstCol1+1:lastCol1-firstCol1+1)<threshold,1,'first');
                rowC = trows1(tcol0+tIndC-1);
                if peroid(rowC)-peroid(trow0) > vCheckFactor*BL_thickness
                    vExtentCheck = 0;
                    break;
                end
                tIndC = find(section2(tcol0-firstCol2+1:lastCol2-firstCol2+1)<threshold,1,'first');
                rowC = trows2(tcol0+tIndC-1);
                if peroid(trow0)-peroid(rowC) > vCheckFactor*BL_thickness
                    vExtentCheck = 0;
                    break;
                end
            end
            
            % check the left side
            angles = pi/16:(atan((trow0-trow)/(rightBound-...
                peakCol(j)))-pi/16)/8:atan((trow0-trow)/(peakCol(j)-leftBound));
            
            for angle = angles
                trows1 = ceil(tan(angle)*((1:length(IM))-tcol0)+trow0);
                trows2 = ceil(-tan(angle)*((1:length(IM))-tcol0)+trow0);
                section1 = zeros(size(wTransform));
                section2 = section1;
                % make matricies that have the line of interest as 1s
                firstCol1 = 1;
                firstCol2 = 1;
                lastCol1 = length(IM);
                lastCol2 = lastCol1;
                for k = 1:length(IM)
                    trow = ceil(tan(angle)*(k-tcol0)+trow0);
                    if trow >0 && trow <= size(section1,1)
                        section1(trow,k) = 1;
                    elseif trow < 1
                        firstCol1 = k+1;
                    elseif trow > size(section1,1) && lastCol1 > k-1
                        lastCol1 = k-1;
                    end
                    trow = ceil(-tan(angle)*(k-tcol0)+trow0);
                    if trow >0 && trow <= size(section2,1)
                        section2(trow,k) = 1;
                    elseif trow > size(section2,1)
                        firstCol2 = k+1;
                    elseif trow < 1 && lastCol2 > k-1
                        lastCol2 = k-1;
                    end
                end
                section1 = abs(wTransform(section1==1));
                section2 = abs(wTransform(section2==1));
                % now section1 is the line going down to the left and section2
                % is the line going up to the left. Check to make sure that
                % the spot still meets specifications in these directions
                tIndC = find(section1(tcol0-firstCol1+1:-1:1)<threshold,1,'first');
                rowC = trows1(tcol0-tIndC+1);
                if peroid(trow0)-peroid(rowC) > vCheckFactor*BL_thickness
                    vExtentCheck = 0;
                    break;
                end
                tIndC = find(section2(tcol0-firstCol2+1:-1:1)<threshold,1,'first');
                rowC = trows2(tcol0-tIndC+1);
                if peroid(rowC)-peroid(trow0) > vCheckFactor*BL_thickness
                    vExtentCheck = 0;
                    break;
                end
            end
            
        end
        
        % Another check to make sure that this is not a turbulent spot
        indC = find(peroid<(peroid(indsN(peakRow(j)))-1.5*BL_thickness),1,'last');
        if isempty(indC)
            indC = 1;
        end
        checkrow = abs(wTransform(indC,leftBound:rightBound));
        threshold = max(minThreshold,peakVal(j)*dropOffPercent*lowerFactor);
        lowerCheck = vExtentCheck && sum(checkrow>threshold)<...
            (lowerPercent*(rightBound-leftBound+1));
        indC = find(peroid>(peroid(indsN(peakRow(j)))+1.5*BL_thickness),1,'first');
        if isempty(indC)
            indC = length(peroid);
        end
        checkrow = abs(wTransform(indC,leftBound:rightBound));
        threshold = max(minThreshold,peakVal(j)*upperFactor*dropOffPercent);
        uperCheck = vExtentCheck && sum(checkrow>threshold)<...
            (upperPercent*(rightBound-leftBound+1));
        
        % check for overlap
        existingPackets = squeeze(wave_packets(i-BL_height+1,:,:));
        existingPackets = existingPackets(existingPackets > 0);
        if sum(sum(existingPackets))==0 %|| min(size(existingPackets))==1
            overlapCheck = 1;
        elseif sum(existingPackets(1,:) < leftBound+overlapFactor*BL_thickness)>0 && ...
                sum(existingPackets(2,:) > rightBound-overlapFactor*BL_thickness)>0
            overlapCheck = 0;
        else
            overlapCheck = 1;
        end
        
        if (rightBound-leftBound > packetLengthFactor*BL_thickness) &&...
                lowerCheck && uperCheck && overlapCheck
            wave_packets(i-BL_height+1,tInd,1) = leftBound;
            wave_packets(i-BL_height+1,tInd,2) = rightBound;
            tInd = tInd+1;
        end
    end
    
    
end


return;


% normalize the weighting
weighting = weighting./max(weighting);

% find the wave packets and their boundaries (same way as
% filteredImageIdentification)
starts = [];
stops = [];
if sum(wave_packets(:,1,1)) > 0
    [starts,stops] = findBounds(wave_packets,weighting,BL_thickness);
    if ~isempty(starts) || ~isempty(stops)
        pairs = pairBounds(starts,stops,wave_packets,BL_thickness,weighting);
        if isempty(pairs)
            starts = [];
            stops = [];
        else
            starts = starts(pairs(:,1));
            stops = stops(pairs(:,2));
            secondMode(loopCount) = 1;
        end
    end
end

% display stuff
if forDisplay
    while ~moveEvent
        displayWavelet(h,Img,image_ft,img_no,cone,BL_height,...
            wave_packets,starts,stops,compiledTransform(indsW,:,:),...
            max(max(max(compiledTransform(indsW,:,:)))),peroid(indsW)./BL_thickness);
        pause();
        loopCount = loopCount+1;
    end
else
    loopCount = loopCount+1;
end

if loopCount < 1
    loopCount = 1;
end











    %function that crops an image to avoid the penumbra transition region
    function crop_col = crop_shadow(img_in)
        img_in = clamp(img_in(1:cone-5,:));
        mean_edge = mean(clamp(sobel(img_in)), 1);
        smooth_diff = smoothdata(diff(mean_edge), 35);
        
        %only consider points between 30 and 400
        smooth_diff(1:30) = 0; smooth_diff(400:end) = 0;
        
        %get the maximum diff for the windowed signal and find the first
        %prominent peak (i.e., where the shadow ends)
        max_val = max(smooth_diff);
        [~, locations] = findpeaks(smooth_diff, 'MinPeakProminence', max_val/2);
        crop_col = locations(1)-10;
        
    end

    %perform the sobel convolution
    function out_img = sobel(in_img)
        v_edge = 1/8*(-in_img(1:end-2,1:end-2)+in_img(3:end,1:end-2)-2*in_img(1:end-2,2:end-1)+2*in_img(3:end,2:end-1)-in_img(1:end-2,3:end)+in_img(3:end,3:end));
        h_edge = 1/8*(in_img(1:end-2,1:end-2)+2*in_img(2:end-1,1:end-2)+in_img(3:end,1:end-2)-in_img(1:end-2,3:end)-2*in_img(2:end-1,3:end)-in_img(3:end,3:end));
        out_img = sqrt(v_edge.^2 + h_edge.^2);
    end




    function out_cell = in_turbulence(curr_packet, turb_train)
        %num_turb = size(turb_train,1);
        m = 1;
        count = 0;
        while m <= size(turb_train,1)
            ii = 1;
            while ii <= size(curr_packet,1)
                count = count + 1;
                if count > 100  %break out of loop if need be
                    m=100;
                end
                if numel(curr_packet)==0   %if there is no packet left
                    continue;
                end
                
                %packet is entirely in turbulence
                if curr_packet(ii,1) >= turb_train(m,1) && curr_packet(ii,2) <= turb_train(m,2)
                    curr_packet(ii,:) = [];
                    %leading edge only in turbulence
                elseif curr_packet(ii,1) >= turb_train(m,1) && curr_packet(ii,1) <= turb_train(m,2)
                    curr_packet(ii,:) = [turb_train(m,2), curr_packet(ii,2), curr_packet(ii,3)];
                    %trailing edge is in packet
                elseif curr_packet(ii,2) >= turb_train(m,1) && curr_packet(ii,2) <= turb_train(m,2)
                    curr_packet(ii,:) = [curr_packet(ii,1), turb_train(m,1), curr_packet(ii,3)];
                %packet contains entire turbulent spot; need to split it
                elseif curr_packet(ii,1) <= turb_train(m,1) && curr_packet(ii,2) >= turb_train(m,2)
                    
                    %pad with zeros to prevent errors
                    curr_packet(end+1,:) = [0,0, curr_packet(ii,3)];
                    
                    %splice new packet into matrix
                    curr_packet(ii:ii+2,:) = [curr_packet(ii,1), turb_train(m,1), curr_packet(ii,3);
                        turb_train(m,2), curr_packet(ii,2), curr_packet(ii,3);
                        curr_packet(ii+1,1), curr_packet(ii+1,2), curr_packet(ii,3)];
                    
                    %remove zero padding
                    curr_packet(end,:) = [];
                    
                
%                 elseif curr_packet(1) <= turb_train(m,1) && curr_packet(2) >= turb_train(m,2)
%                     turb_train(m, :) = [];
%                     m = m - 1;
%                     continue;
                end
                ii = ii + 1;
            end
            m = m + 1;
        end
        new_packet = curr_packet;
        out_cell = {new_packet; turb_train};
    end





end