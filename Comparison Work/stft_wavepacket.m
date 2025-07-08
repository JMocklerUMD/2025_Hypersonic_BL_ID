function [second_mode, packets] = stft_wavepacket(shot, img_num,...
    no_ref, furthest_image, forDisplay, turbulenceInfo, turbulent1)
%[ turbulent, spots] = stft_turbulence(shot, img_num, no_ref, furthest_image, forDisplay, turbulenceInfo, turbulent1)
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
%   Function to find turbulent spots by means of an stft
%
%   Code adapted from intermittency routine



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


fprintf('%d -- [%d %d]\n', img_num, size(ref_img, 1), size(ref_img, 2));


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
    intensity_curve = smoothdata(mean(ref_img(cone-7:cone-2,:),1),51);
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


turb_rows = BL_height - floor(BL_thickness/2):cone;
j = 1;

%9*delta as always
window_size = 12*BL_thickness;

%compute the STFT spectrogram in each row
for row = turb_rows
    section = curr_img_filt(row,:);
    [s, f, t] = spectrogram(section, hamming(window_size), ...
        ceil(window_size*0.8), window_size, BL_thickness, 'yaxis');
    spec(:,:,j) = s;
    j = j + 1;
end

%average the spectra of all rows to reduce the variance 
spec = mean(abs(spec).^2,3);


turb_freq_band = floor([2/BL_thickness;3.5/BL_thickness] * window_size);
turb_freq_band = turb_freq_band(1):turb_freq_band(2);

%use round rather than floor for second-mode since range is so small
second_mode_freq_band = round([1/(3.5*BL_thickness);1/(2*BL_thickness)] * window_size);
second_mode_freq_band = second_mode_freq_band(1):second_mode_freq_band(2);


%change these values for the appropriate shots
if shot == 1296
    turb_thresh  = 2.9e-03;
    power_thresh = 65;
elseif shot == 1297
    turb_thresh  = 4.5e-03;
    power_thresh = 43;
elseif shot == 4016
    turb_thresh = 1e-03;
    power_thresh = 4e-02;
elseif shot == 4017
    turb_thresh = 1.8e-3;
    power_thresh = 0.16;
elseif shot == 4119  %need to change these when running for 4119
    turb_thresh = 1.3248e-04*1.0;
    power_thresh = 0.0019*1.7;
end

    turb_thresh = 1;
    power_thresh = 0.5;


%get the turbulent spots
turb_spectrum_sum = mean(spec(turb_freq_band,:),1);
turbulent = turb_spectrum_sum > turb_thresh;

%then the second-mode wave-packets
second_mode_spectrum_sum = mean(spec(second_mode_freq_band,:),1);

%use the ratio instead for HEG shots
if shot == 1296 || shot == 1297
   second_mode_spectrum_sum = second_mode_spectrum_sum./turb_spectrum_sum; 
end

second_mode = abs(second_mode_spectrum_sum) > power_thresh;



% %UNCOMMENT FOR TESTING
% figure(1);
% plot(smooth(abs(turb_spectrum_sum),5));
% figure(2);
% imshow(clamp(curr_img_filt));
 



step_size = floor(diff(t(1:2))*BL_thickness);

%form packets out of adjoining turbulent spots/wave-packets
train = cell(2,1);
train{1} = zeros(10,2); train{2} = zeros(10,2); train{3} = zeros(10,2); %set to 10 to supress warnings

wave_packets{1} = second_mode;
wave_packets{2} = [];
wave_packets{3} = turbulent;  %this is a hack to concatenate turb spots

for i = [1,2,3]
    wave_packet_frames = find(wave_packets{i});
    train_num = 1;
    if length(wave_packet_frames) > 1
        wave_packet_diff = diff(wave_packet_frames);
        for curr_packet = 1:length(wave_packet_frames)
            
            %if we don't have a starting point for the current wave train
            if train{i}(train_num, 1) == 0
                train{i}(train_num, 1) = (wave_packet_frames(curr_packet)-1)*step_size+1;
            end
            
            %if the next wave packet is less than four steps away, it is in
            %the same wave train so add it to the end
            if curr_packet == length(wave_packet_frames)  %if this is the last packet it has to end the train
                train{i}(train_num, 2) = (wave_packet_frames(curr_packet)-1)*step_size+window_size;
            elseif wave_packet_diff(curr_packet) < 5
                train{i}(train_num, 2) = (wave_packet_frames(curr_packet+1)-1)*step_size+window_size;
            else %otherwise this packet ends the train and we iterate
                train{i}(train_num, 2) = (wave_packet_frames(curr_packet)-1)*step_size+window_size;
                train_num = train_num + 1;
            end
            
        end
    elseif isscalar(wave_packet_frames) && wave_packet_frames ~= 0
        train{i}(train_num, 1) = (wave_packet_frames-1)*step_size+1;
        train{i}(train_num, 2) = (wave_packet_frames-1)*step_size+window_size;
    end
    train{i} = reshape(train{i}(find(train{i})), length(find(train{i}))/2, 2); %get rid of the pre-allocated zeros
end




%adjust wavepacket locations for turbulence continuity
for i = 1
    train_num = 1;
    train_size = size(train{i},1);
    while train_num <= size(train{i},1)
        packet_cell = in_turbulence(train{i}(train_num,:), train{3});
        %[train{i}(train_num,:), train{3}] = in_turbulence(train{i}(train_num,:), train{3});
        if numel(packet_cell{1,1}) == 2
            train{i}(train_num,:) = packet_cell{1,1};
        elseif numel(packet_cell{1,1}) == 0
            train{i}(train_num,:) = [];
        else  %need to consider case in which wavepackets split into 3 parts
            train{i}(end+1,:) = [0,0];
            train{i}(train_num:end+size(packet_cell{i,1},1)-1,:) = [packet_cell{i,1};
                train{i}(train_num+1:end,:)];
            train{i}(end,:) = [];
        end
        
        if (packet_cell{2,1})
            train{3} = packet_cell{2,1};
        else
            train{3} = [];
        end
        
        %see if we reduce the number of packets
        if size(train{i},1) < train_size
            train_size = size(train{i},1);
        else
            train_num = train_num + 1;
        end
        
    end
    
end


    function out_cell = in_turbulence(curr_packet, turb_train)
        %num_turb = size(turb_train,1);
        m = 1;
        count = 0;
        while m <= size(turb_train,1)
            ii = 1;
            while ii <= size(curr_packet,1)
                count = count + 1;
                if count > 20  %break out of loop if need be
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
                    curr_packet(ii,:) = [turb_train(m,2), curr_packet(ii,2)];
                    %trailing edge is in packet
                elseif curr_packet(ii,2) >= turb_train(m,1) && curr_packet(ii,2) <= turb_train(m,2)
                    curr_packet(ii,:) = [curr_packet(ii,1), turb_train(m,1)];
                %packet contains entire turbulent spot; need to split it
                elseif curr_packet(ii,1) <= turb_train(m,1) && curr_packet(ii,2) >= turb_train(m,2)
                    
                    %pad with zeros to prevent errors
                    curr_packet(end+1,:) = [0,0];
                    
                    %splice new packet into matrix
                    curr_packet(ii:ii+2,:) = [curr_packet(ii,1), turb_train(m,1);
                        turb_train(m,2), curr_packet(ii,2);
                        curr_packet(ii+1,1), curr_packet(ii+1,2)];
                    
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

    

%check the width of the wavepackets; they should be at least three step
%sizes wide to be considered full packets rather than anomalous positives
for i = 1:2
    if size(train{i},1) > 0 
        num_packets = size(train{i},1);
        kk = 1;
        for k = 1:num_packets
            if diff(train{i}(kk,:)) < 15*BL_thickness
                train{i}(kk,:) = [];
            else
                kk = kk + 1;
            end
        end
        
    end
end


%calculate center of bins for plotting ratio
% mode_locations = (1:length(mode_ratios))*step_size+step_size/2;
% 

if isnumeric(shot)
    shot = int2str(shot);
end



%get number of each type for plotting purposes
num_weak = size(train{1},1);
num_strong = size(train{2},1);
num_turb = size(train{3},1);

weak_colors = cell(num_weak+num_turb,1);
weak_colors(1:num_weak) = {'second'};
weak_colors(num_weak+1:end) = {'turb'};

strong_colors = cell(num_strong+num_turb,1);
strong_colors(1:num_strong) = {'second'};
strong_colors(num_strong+1:end) = {'turb'};




% %UNCOMMENT ME FOR TESTING
% fig = figure(2);
% subplot(2, 1, 1);
% if (~isempty(train{3}))
%     quick_view(curr_img_filt, [train{1}(:,1); train{3}(:,1)],...
%         [train{1}(:,2); train{3}(:,2)], fig, 0, weak_colors);
% else
%     quick_view(curr_img_filt, train{1}(:,1), train{1}(:,2), fig, 0, weak_colors);
% end
% header = sprintf('MUSIC Decomposition: %d weak packet(s) (Shot %s, Image %d)', size(train{1},1), shot, img_num);
% title(header);
% set(fig, 'position', [100 500 1800 650]);
% 
% subplot(2, 1, 2);
% if (~isempty(train{3}))
%     quick_view(curr_img_filt, [train{2}(:,1); train{3}(:,1)],...
%         [train{2}(:,2); train{3}(:,2)], fig, 0, strong_colors);
% else
%     quick_view(curr_img_filt, train{2}(:,1), train{2}(:,2), fig, 0, strong_colors);
% end
% header = sprintf('MUSIC Decomposition: %d strong packet(s) (Shot %s, Image %d)', size(train{2},1), shot, img_num);
% title(header);




packets = cell(3, 1);
second_mode = [0; 0];
%prepare output
for i = [1, 2, 3]
    if (size(train{i},1) > 0)
        second_mode(i) = 1;
        packets{i} = [train{i} img_num*ones(size(train{i}, 1), 1)];
        %if there is a wavepacket, also output the start and stop locations
    else
        second_mode(i) = 0;
    end
end



end
