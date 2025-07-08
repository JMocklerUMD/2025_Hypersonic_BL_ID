function [ second_mode, packets] = music_identification_v4(shot, img_num, no_ref)
%
%[second_mode, turbulent_imgs ] = music_identification(shot, loopRange, no_ref, furthest_image, forDisplay, turbulenceInfo, turbulent1)UNTITLED2 Summary of this function goes here
% 
%   Inputs:
%       shot - the number of the shot that you are interested in working
%           with.
%       img_num - number of the current image from the converted tiff files
%       no_ref - the number of reference images, from each side, to be
%           used to construct the average reference image
%
%   Outputs:
%       second_mode - the images which had identified 2nd mode wave packets
%           in them
%       packets - matrix of pixel limits and img number of wave-packets
%
%   This function determines whether the current frame contains second mode
%   wave packets based on an estimate of its pseudospectrum (using the
%   multiple signal classification MUSIC method). If the eigenvalue
%   representing a second mode packet is prominent (above a certain
%   predetermined or adaptive threshold), then the current image will
%   be marked as containing a second mode packet.  
%
%   This function also has the capability of identifying turbulent spots.
%   At the most naive level, this would be marked by an elevated relative
%   intensity of high frequency terms in the pseuospectrum (as opposed to
%   high frequecy noise, which will not be significant).
%

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
%addpath([pwd '/wave_packet_detection'])
base_dir = pwd;

% get the correct function to read the images with
Langley_shots = [2:81];
if sum(shot==Langley_shots)>0
    imgReadFxn = findImageReadFxn(shot);
    base_dir_Langley = 'C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\Comparison Work\\Run33_data';
    curr_img = imgReadFxn(base_dir_Langley,shot,img_num);
    curr_img = double(curr_img);
else
    imgReadFxn = findImageReadFxn(shot);
    curr_img = imgReadFxn(base_dir,shot,img_num);
    curr_img = double(curr_img);
end


% first, get a reference image to build off of (and save some computations)
% this is the averaged image for 2*no_ref+1 images centered on 'img_no'
if regexp(sprintf('%s',shot), 'dmd')
    ref_img = imgReadFxn(base_dir,shot,0);
    ref_img = double(ref_img);
elseif sum(shot==Langley_shots)>0
    base_dir_Langley = 'C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\Comparison Work\\Run33_data';
    [ref_img,nnPrev] = referenceImage(no_ref,stop,img_num,base_dir_Langley,shot,...
        0, 0, 0);
else
    [ref_img,nnPrev] = referenceImage(no_ref,stop,img_num,base_dir,shot,...
        0, 0, 0);
end
   





%divide images by the maximum pixel values in order to standardize the
%PSD values encountererd when performing spectral analysis

info = imfinfo([img_dir '\' eval(file_name)]);
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
    if bl_invert  %invert the image for bl finding if necessary
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




%%%%%%%%%%%%%%
%Filtering in preparation for spectral extraction
%%%%%%%%%%%%%%

%design band-pass filter to focus on the second-mode wavelengths
%band limits are 1/(3*delta) and 1/(2*delta)
% bp_filter1 = fdesign.bandpass(1/(4.5*BL_thickness), (1.5)/(4.5*BL_thickness),...
%     2/(1*BL_thickness), (3.2)/(1.5*BL_thickness), 5, 2, 5);
bp_filter1 = fdesign.bandpass(1/(9*BL_thickness), (2)/(9*BL_thickness),...
    12/(9*BL_thickness), (15)/(9*BL_thickness), 5, 2, 5);

%temporary filter for testing
if shot == 1296 || shot == 4017 
    bp_filter1 = fdesign.bandpass(2/(9*BL_thickness), (3)/(9*BL_thickness),...
        18/(9*BL_thickness), (21)/(9*BL_thickness), 5, 2, 5);
end

bp_filter1 = design(bp_filter1,'equiripple');
bp_filter1_scale = 0.1297;

%filter for first harmonic
bp_filter2 = fdesign.bandpass(2/(1.5*BL_thickness),(2.2)/(1.5*BL_thickness),...
    2/BL_thickness,2.2/BL_thickness,5,2,5);
bp_filter2 = design(bp_filter2,'equiripple');

%filter for turbulence (limited high-pass)
bp_filter3 = fdesign.bandpass(1.4/BL_thickness,1.5/BL_thickness,...
    4/BL_thickness,4.1/BL_thickness,5,2,5); 
% bp_filter3 = fdesign.bandpass(1.4/BL_thickness,1.5/BL_thickness,...
%     20/BL_thickness,20.1/BL_thickness,5,2,5); 
bp_filter3 = design(bp_filter3,'equiripple');


%preallocate memory for code optimization
pad=100;
filtered_img1 = zeros(cone - BL_height+1,img_length+pad);
filtered_img2 = zeros(cone - BL_height+1,img_length+pad);
filtered_img3 = zeros(cone,img_length+pad); %turbulence not constrained to BL

%perform the aforementioned filtering to obtain band-passed images
%zero pad the input image so that no info is lost due to filter signal shift
length_img = size(curr_img_filt, 2);
pad_img = zeros(size(curr_img_filt,1), pad);
for i = 1:cone
    tmp_img = [curr_img_filt, pad_img];
    if i >= BL_height

        filtered_img1(i-BL_height+1,:) = filter(bp_filter1,tmp_img(i,:));
        filtered_img2(i-BL_height+1,:) = filter(bp_filter2,tmp_img(i,:));
    end
    filtered_img3(i,:) = filter(bp_filter3,tmp_img(i,:));
end


%take the cross-correlation of the second-mode image (essentially low-pass)
%and apply the lag to all filtered images. This won't be exactly the same,
%but it is nearly impossible to accurately gauge the shift in the higher
%frequency images
%take this at three points in the boundary layer and average
cor_p = floor(BL_thickness./[1, 2, BL_thickness]);
[cor1, lag] = xcorr(filtered_img1(end-cor_p(1),:), curr_img_filt(cone-cor_p(1),:));
[cor2, ~] = xcorr(filtered_img1(end-cor_p(2),:), curr_img_filt(cone-cor_p(2),:));
[cor3, ~] = xcorr(filtered_img1(end-cor_p(3),:), curr_img_filt(cone-cor_p(3),:));

lag1 = lag(find(cor1==max(cor1)));
lag2 = lag(find(cor2==max(cor2)));
lag3 = lag(find(cor3==max(cor3)));
lag = mean([lag1, lag2, lag3]);
if (lag+length_img-1 > size(filtered_img1, 2))
    lag = 50;  %if lag calculations incorrect, set to 50 (good approx for filter order)
end
lag = ceil(lag);
filtered_img1 = filtered_img1(:,lag:lag+length_img-1);
filtered_img2 = filtered_img2(:,lag:lag+length_img-1);
filtered_img3 = filtered_img3(:,lag:lag+length_img-1);







%%%%%%%%%%%%%%
%Pseudo-spectrum calculation
%%%%%%%%%%%%%%

%%%%%%current methodology
%for now, let's do this row-wise
%for each window find the ratio of the peak 2-nd harmonic intensity to
%  the mean turbulence wavelength density
%each window is the length of 3 wavepackets
%the number of wavepackets is determined by the number of steps above
%  a given ratio (which is yet to be determined)


%initialize relevant variables
step_size = 3 * BL_thickness;
window_size = 3*step_size;
%num_steps = 7+3*(floor(imgLength/window_size)-3);  %sliding window length
num_steps = (floor(img_length/step_size)) - 2;  %sliding window length
mack_band = ceil(window_size/(BL_thickness*3.5)):1:floor(window_size/(BL_thickness*1.5));

%%%%%%%%%%%%%%%%%%%%%%
%DETECTION THRESHOLDS%
%%%%%%%%%%%%%%%%%%%%%%
eval(['peak_range =' peak_range ';']);

%preallocate space for efficiency
nfft = window_size;
wave_packets = cell(2,1);
turb_spots = zeros(1,num_steps);

for step_num = 0:num_steps-1
    
    %obtain the proper section of the band-passed image and run music
    section = filtered_img1(1:BL_thickness, step_num*step_size+(1:window_size));
    
    %NOTE: should not be using the filtered image for pseudospectral estimation;
    %this is basically forcing the spectrum to appear a certain way
%     section = curr_img_filt(1:BL_thickness, step_num*step_size+(1:window_size));

    [curr_music_spec, peak_freq, peak_pow] = music_spec_avg(section, 3, size((section),2)-1);
    peak_freq = peak_freq/8192*window_size;
    
%     semilogy([1:600]/8192*window_size, curr_music_spec(1:600)) 
    
    
    
    
    %turb_method = 'canny'
    %turb_thresh = 0.2;
    
    
    %now determine the turbulence method to use
    switch turb_method

        %highf turbulence test checks for prominent signals at high
        %frequency; signal is modified by a hamming window
        case 'highf'
            turb_rows = floor(linspace(BL_height - floor(BL_thickness/2), cone, 6));
            turb_rows = BL_height - floor(BL_thickness/2):cone;
            k = 1;
            for row = turb_rows
                %section = filtered_img3(row, step_num*step_size+(1:window_size)).* hamming(window_size)';
                %section = curr_img_filt(row, step_num*step_size+(1:window_size)).* hamming(window_size)';
                %section = filtered_img3(row, step_num*step_size+(1:window_size));
                section = curr_img_filt(row, step_num*step_size+(1:window_size));
                %curr_spectrum = music_spec(section, 30, numel(section)-1);
                %[curr_spectrum, curr_freq] = pmusic(section, [15, 1], nfft, window_size);
                [curr_spectrum, curr_freq] = periodogram(section, hamming(window_size), window_size, 1);
                
                %if we have uncorrected optical artifacts, the spectrum
                %will be normalized much higher, so throw away that
                %spectrum if the mean is above a certain value
                test_turb_range = 13:30;
                test_thresh = 3e-4;
                if shot == 1296 || shot == 1297
                    %the test thresh was used to exclude imperfections in the image
                    %but 1296 just has strong turbulence so ignore this check
                    test_thresh = 1000;
                end
                
                if mean(curr_spectrum(test_turb_range)) > test_thresh
                    k = k-1;
                else
                    turb_spectrum(k,:) = curr_spectrum;
                end
                %turb_freq(1,:) = curr_freq;
                
                
                %for paper figure we want to show a spectrogram plot so
                %store these values for each row then average
%                 section = curr_img_filt(row,:);
%                 [s, f, t] = spectrogram(section, hamming(window_size), ...
%                     ceil(window_size*0.95), window_size, BL_thickness, 'yaxis');
%                 spec(:,:,k) = s;
%                 
%                 
                k = k + 1;
                    
            end
            
%             spec = mean(spec,3);
            print_spec = 0;   %revisit this at some later point
            if print_spec
                
                f = 2*f;
                
                figure(1);
                imagesc(t*BL_thickness,f,log10(abs(spec).^2));
                xlabel('Location Along Image [Pixels]');
                ylabel('Wavenumber [1/$\delta$]', 'interpreter', 'latex');
                ylabel('Wavenumber [$k_{2^\mathrm{nd}}$]', 'interpreter', 'latex');
                h = colorbar;
                colormap(hot);
                %caxis([-5 max(log10(abs(spec(:)).^2))]);
                ylabel(h, 'log(Power Spectral Density)');
                set(gca, 'ydir', 'normal');
                set(gca, 'fontsize', 14);
                set(gcf, 'Position', [300, 300, 900, 500]);
                
                %now compare spectra at laminar/turbulent positions
                % figure(2);
                % s1 = abs(spec(:,5)).^2;
                % s2 = abs(spec(:,55)).^2;
                % semilogy(f, smooth(s1,5), 'linewidth', 1.5);
                % hold on;
                % semilogy(f, smooth(s2,5), '--', 'linewidth', 1.5);
                % semilogy([3.5, 3.5], [10^-6 1], '-.k', 'linewidth', 1.5);
                % semilogy([7, 7], [10^-6 1], '-.k', 'linewidth', 1.5);
                % grid on;
                % xlabel('Wavenumber [1/$\delta$]', 'interpreter', 'latex');
                % xlabel('Wavenumber [$k_{2^\mathrm{nd}}$]', 'interpreter', 'latex');
                % ylabel('Power Spectral Density');
                % legend('Transitional Region', 'Turbulent Region');
                % set(gca, 'fontsize', 14);
                % set(gcf, 'Position', [300, 300, 900, 500]);
                % 
            end
            
            
            
            turb_mean_spectrum = mean(turb_spectrum);
            if shot == 4016
                turb_freq_band = floor([1.7/BL_thickness;4/BL_thickness] * 9*BL_thickness);
                turb_freq_band = turb_freq_band(1):turb_freq_band(2);
            elseif shot == 4017
                turb_freq_band = 17:30;
            %elseif shot == 1296
                turb_freq_band = 17:35;
%                 turb_freq_band = floor([1.5/BL_thickness;4/BL_thickness] * 9*BL_thickness);            
%                 turb_freq_band = turb_freq_band(1):turb_freq_band(2);
            elseif shot == 4119
                turb_freq_band = 15:25;
                turb_freq_band = 18:31;
            elseif shot == 4118
                turb_freq_band = floor([2/BL_thickness;3.5/BL_thickness] * 9*BL_thickness);
                turb_freq_band = turb_freq_band(1):turb_freq_band(2);
            else
                turb_freq_band = floor([1.5/BL_thickness;4/BL_thickness] * 9*BL_thickness);
                turb_freq_band = floor([2/BL_thickness;3.5/BL_thickness] * 9*BL_thickness);
                turb_freq_band = floor([1.5/BL_thickness;3.5/BL_thickness] * 9*BL_thickness);
                turb_freq_band = turb_freq_band(1):turb_freq_band(2);
            end
            
            turb_freq_band = floor([2/BL_thickness;3.5/BL_thickness] * 9*BL_thickness);
            turb_freq_band = turb_freq_band(1):turb_freq_band(2);
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %table of threshold values for strong waves and turbulence
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if shot == 1296
                turb_thresh  = 1.5242e-04*0.687;
%                 power_thresh = 8.6026e-05*1.67;
%                  turb_thresh  = 1.5242e-04*0.66;
                 power_thresh = 8.6026e-05*0.7;   %edited after 1296 bl adjustment
            elseif shot == 1297
                turb_thresh  = 1.2052e-04*0.83;
%                 power_thresh = 3.2341e-05*2.6;
                turb_thresh  = 1.5242e-04*0.75;
                power_thresh = 8.6026e-05*0.3;   %edited after 1297 bl adjustment
            elseif shot == 4016
                turb_thresh = 1.1475e-05*2.5;
                power_thresh = 4.0197e-05*0.7;
            elseif shot == 4017
                turb_thresh = 2.4162e-05*1.1;
                power_thresh = 2.7651e-05*1.55;
            elseif shot == 4119
                turb_thresh = 1.3248e-04*1.0;
                power_thresh = 0.0019*1.7;
            end
            
            
            %changed the turbulent summation method to mean;
            %will prevent a situation in which round-off error creates a
            %disparity in the size of the summation limits
%             figure(1);semilogy(smooth(turb_mean_spectrum(:),7));hold on;
            turb_spectrum_sum = sum(turb_mean_spectrum(turb_freq_band));
            turb_spectrum_sum = mean(turb_mean_spectrum(turb_freq_band));
            %test section
            section = filtered_img3(turb_rows, step_num*step_size+(1:window_size));
            %test = music_spec_avg(section, 5, size((section),2)-1);
            %semilogy((1:1000)/8192*window_size,test(1:1000));
            
            
            %test for turbulence
            turbulent = turb_spectrum_sum > turb_thresh;
            
            %include extra check for higher 2nd mode harmonics by computing
            %column-wise FFT
            section = curr_img_filt(turb_rows, step_num*step_size+(1:window_size));
            section = section(:);     %use linear index convention rather than sub-index
            
%             [col_turb_spectrum, f] = periodogram(section, [], [], 2*BL_thickness);
%             figure(1);
%             semilogy(f(1:end-100), ...
%                 smooth(col_turb_spectrum(1:end-100),31));
%             hold on;
%             grid on;
            
            %test the use of canny edge with high f as a check on harmonics
            canny_edges = edge(curr_img_filt,'canny', 0.275);
            section = canny_edges(1:BL_height-4, step_num*step_size+(1:window_size));
%            turbulent = turbulent_section(section) && turbulent;

            canny_edges = double(canny_edges);
            canny_edges(BL_height-4,:) = 0.2;
%             figure(3);
%            imshow(canny_edges);
           
            
        %perform canny edge detection on the current section
        case 'canny'
            canny_edges = edge(curr_img_filt,'canny', turb_thresh);
            section = canny_edges(1:BL_height-4, step_num*step_size+(1:window_size));
            turbulent = turbulent_section(section);
            
    end
    
    
    %%%%UNCOMMENT ME FOR TESTING
%     fig = figure(2);
%     quick_view(curr_img_filt, step_num*step_size+1, step_num*step_size+window_size, fig);
    
    %mode_ratio_string = sprintf('Mack Norm: %d, Mack Max %d, Turb Norm: %d', round(mack_mean_spectrum), round(mack_max_spectrum), round(turb_mean_spectrum));
    %title(mode_ratio_string);
    
    
    
    %if we have a peak in the denoted range, there is a wave packet present
    second_mode_peak = min(find(peak_freq >= peak_range(1) & peak_freq <= peak_range(2)));
    if numel(second_mode_peak) > 0 && ~turbulent
        wave_packets{1}(step_num + 1) = 1;
        
        %now determine if this is also a strong packet
        if peak_pow(second_mode_peak) > power_thresh
            wave_packets{2}(step_num + 1) = 1;
        else
            wave_packets{2}(step_num + 1) = 0;
        end
    else
        wave_packets{1}(step_num + 1) = 0;
        wave_packets{2}(step_num + 1) = 0;
    end
    
    
    %denote turbulent spots
    if turbulent
        turb_spots(step_num + 1) = 1;
    end
    
    
     continue;
    
    %now that we'vthresh = mean(mean_thresh) - std(mean_thresh);e obtained the row-averaged band-limited noise-reduced pseudospectrum
    %  get the mean spectrum in the expected second mode band
    %    max_freq = freq(find(mean_spectrum==max(mean_spectrum)));
    turb_freq_band = floor([1.5/BL_thickness;4/BL_thickness] * 9*BL_thickness);
    mack_mean_spectrum = mean(mean_spectrum(mack_band));
    mack_max_spectrum = max(mean_spectrum(mack_band));
    turb_mean_spectrum = mean(turb_mean_spectrum(turb_freq_band));

   %mode_ratio = mack_mean_spectrum/turb_mean_spectrum;
   mode_ratio = mack_max_spectrum/turb_mean_spectrum;

   mode_ratios(step_num+1) = mode_ratio;
   mack_norms(step_num+1)  = mack_mean_spectrum;
   mack_maxes(step_num+1)  = mack_max_spectrum;
   turb_norms(step_num+1) =  turb_mean_spectrum;



   %run the tests on the previous, not current window (once it is averaged)
%    switch is_turbulent
%        case 0 
%            test_results = (mode_ratio > ratio_threshold);
%        case 1
%            %test_results = ((mack_mean_spectrum > mack_threshold || mack_max_spectrum > mack_max_threshold) && turb_mean_spectrum< turb_threshold);
%            test_results = (mack_max_spectrum > mack_max_threshold && turb_mean_spectrum< turb_threshold);
%    end
       
%%%%%%%%%
%previous tests for wavepackets
%   if (mode_ratio > ratio_threshold && turb_mean_spectrum< turb_threshold)
%   if ((mack_mean_spectrum > mack_threshold || mack_max_spectrum > mack_max_threshold) && turb_mean_spectrum< turb_threshold)
%    if (test_results)
%        wave_packets(step_num+1) = 1;
%        mode_ratio_string = sprintf('Wavepacket confirmed; Mack Norm: %d, Mack Max %d, Turb Norm: %d', round(mack_mean_spectrum), round(mack_max_spectrum), round(turb_mean_spectrum));
%    else
%        mode_ratio_string = sprintf('No Wavepacket; Mack Norm: %d, Mack Max: %d, Turb Norm: %d', round(mack_mean_spectrum), round(mack_max_spectrum), round(turb_mean_spectrum));
%    end
%    
   fig = figure(2);
   quick_view(curr_img_filt, step_num*step_size+1, step_num*step_size+window_size, fig);
   mode_ratio_string = sprintf('Mack Norm: %d, Mack Max %d, Turb Norm: %d', round(mack_mean_spectrum), round(mack_max_spectrum), round(turb_mean_spectrum));
   title(mode_ratio_string);
% 
%    figure(1);
%    plot(1:numel(mean_spectrum), mean_spectrum);
   
%  pause(0.5);
end


%do the post-processing convolution and detection here
avg_arr = 1/9.0*[1, 2, 3, 2, 1];
for i = [1, 2]
    continue
    switch is_turbulent
        case 0
            %during convolution pad the beginning and end with the first and
            %last values, maybe write a function to do this
            mode_ratios_new = conv([mode_ratios(1),mode_ratios(1),mode_ratios',mode_ratios(end),...
                mode_ratios(end)], avg_arr, 'same');
            mode_ratios_new = mode_ratios_new(3:end-2);
            wave_packets{i} = mode_ratios_new > ratio_threshold(i);
        case 1
            mack_maxes = conv(mack_maxes, avg_arr, 'same');
            %        turb_norms = conv(turb_norms, avg_arr, 'same');
            wave_packets{i} = (mack_maxes > mack_max_threshold(i)) .* (turb_norms< turb_threshold(i));
    end
end

    



%now find wavepacket trains by determining whether they are consecutive
train = cell(2,1);
train{1} = zeros(10,2); train{2} = zeros(10,2); train{3} = zeros(10,2); %set to 10 to supress warnings

wave_packets{3} = turb_spots;  %this is a hack to concatenate turb spots

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
for i = [1,2]
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
            train{i}(train_num:end+size(packet_cell{1,1},1)-1,:) = [packet_cell{1,1};
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
        if shot == 4017 || shot == 4016
            num_waves = 3;
        else
            num_waves = 4;
            num_waves = 7.5;   %ok, let's use 7.5 here since the results were oddly much better
       end
        kk = 1;
        for k = 1:num_packets
            if diff(train{i}(kk,:)) < num_waves*2*BL_thickness
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


%{
%UNCOMMENT ME FOR TESTING
fig = figure(2);
subplot(2, 1, 1);

if (~isempty(train{3}))
    quick_view(curr_img_filt, [train{1}(:,1); train{3}(:,1)],...
        [train{1}(:,2); train{3}(:,2)], fig, 0, weak_colors);
else
    quick_view(curr_img_filt, train{1}(:,1), train{1}(:,2), fig, 0, weak_colors);
end
header = sprintf('MUSIC Decomposition: %d weak packet(s) (Shot %s, Image %d)', size(train{1},1), shot, img_num);
title(header);
set(fig, 'position', [100 500 1800 650]);

subplot(2, 1, 2);
if (~isempty(train{3}))
    quick_view(curr_img_filt, [train{2}(:,1); train{3}(:,1)],...
        [train{2}(:,2); train{3}(:,2)], fig, 0, strong_colors);
else
    quick_view(curr_img_filt, train{2}(:,1), train{2}(:,2), fig, 0, strong_colors);
end
header = sprintf('MUSIC Decomposition: %d strong packet(s) (Shot %s, Image %d)', size(train{2},1), shot, img_num);
title(header);
%}



% plot(mode_locations, mode_ratios);
% xlabel('Location Along Cone Surface (Pixels)');
% ylabel('2nd-mode Ratio');
 

 %subplot(3,1,3);
 %imshow(canny_turb(1:BL_height-4, :));

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
return;


    %function for padded convolution with kernel of length 5
    function conv_vec = pad_conv(vec, avg_vec)
        if (numel(avg_vec) ~= 5)
            error('Must have smoothing kernel of length 5.');
        end
        
        %determine padding length
        pad_length = floor(numel(avg_vec)/2);
        conv_vec = conv(vec, avg_vec, 'same');
        for curr_element = 1:pad_length
            for conv_element = 1:pad_length
                
                
            end
        end
        
    end


    function spec_out = music_spec(sig_in, p, M)
        %get the covariance matrix and perform eigen-analysis
        [~,Rx] = corrmtx(sig_in, M);
        [eig_vec, eig_val] = eig(Rx);
        
        %sort the eigenvectors
        [~, new_ind] = sort(eig_val);
        eig_vec = eig_vec(:,new_ind);
        
        %now sum the fft of the eigenvectors and invert
        spec_out = 0;
        for n_vec = 1:M-p
           spec_out = spec_out + abs(fft(eig_vec(:,n_vec), 8192)); 
        end
        spec_out = spec_out.^(-2);
        %spec_out = 10*log10(spec_out);
        
    end


    %variation on music spectrum identification whereby we average the
    %autocorrelation matrices for the selected rows in the band-filtered
    %image (rather than averaging the output spectrum)
    function [spec_out, peak_freqs, peak_pow] = music_spec_avg(sig_in, p, M)
        
        Rx = zeros( M+1, M+1);
        %get the autocorrelation matrix for each row
        for j = 1:size(sig_in ,1)
            [~,Rx_curr] = corrmtx(sig_in(j,:), M);
            Rx = Rx + Rx_curr;
        end
        
        %get the average autocorrelation matrix and 
        Rx = Rx/size(sig_in ,1);
        [eig_vec, eig_val] = eig(Rx);

        %sort the eigenvectors
        [eig_val, new_ind] = sort(diag(eig_val), 'ascend');
        eig_vec = eig_vec(:,new_ind);
        
        %now sum the fft of the eigenvectors and invert
        spec_out = 0;
        for n_vec = 1:M-p
           spec_out = spec_out + abs(fft(eig_vec(:,n_vec), 8192)); 
        end
        spec_out = spec_out.^(-2);
        %spec_out = 10*log10(spec_out);
        
        
        
        
        
        %%%%%%%reconstruct the power of the complex exponentials
        freqs = linspace(0,2*pi, numel(spec_out));
        temp_spec = spec_out(1:ceil(numel(spec_out)/2));
        freqs = freqs(1:ceil(numel(spec_out)/2));
        
        %get the frequencies of the 'p-1' largest peaks
        %since this isn't zero-mean, we use w=0 as the most prominent
        %frequency in the signal, which is accurate to a good approximation
        [peaks, locations] = findpeaks(temp_spec, 'MinPeakProminence', 1e-3);
        [~, peak_ind] = sort(peaks, 'descend');
        
        %if there are fewer signals than prominent peaks
        if p > numel(peaks)
            p = numel(peaks);
        end
        
        peak_freqs = [1; locations(peak_ind(1:p-1))];
        peak_freqs = [locations(peak_ind(1:p))];

        [eig_val, new_ind] = sort(eig_val, 'descend');
        eig_vec = eig_vec(:,new_ind);
        
        %now solve the linear equations for the power of the sinusoids
        pow_mat = zeros(p);
        %figure(1);
        for n_vec = 1:p
            curr_spec = fft(eig_vec(:,n_vec), 8192);
            pow_mat(n_vec,:) = abs(curr_spec(peak_freqs)).^2;
            %subplot(3, 1, n_vec);
            %plot([1:600]/8192*window_size, abs(curr_spec(1:600)).^2);
        end
        
        a = eig_val(1:p) - eig_val(end);
        peak_pow = pow_mat\a;
        
        peak_pow = abs(peak_pow);
        
        
        pseudo_plot = 0;
        if pseudo_plot
            freqs = (1:numel(temp_spec))/8192*window_size;
            semilogy(freqs/4.5, temp_spec)
            hold on;
            plot([4/7, 4/7], [10^0, 10^-4], '--k', 'linewidth', 1.5)
            plot([1, 1], [10^0, 10^-4], '--k', 'linewidth', 1.5)
            xlim([0 3])
            xlabel('Wavenumber [1/9 $\delta$]', 'interpreter', 'latex');
            xlabel('Wavenumber [1/9$\delta$]', 'interpreter', 'latex');
            ylabel('Spectral Density', 'interpreter', 'latex')
            plot(peak_freqs/4.5/8192*window_size, peaks(peak_ind(1:p)), 'ro');
            set(gca, 'fontsize', 15);
            set(gcf, 'position', [400 500 900 800]);
            hold off;
        end
        
    end




    %function to determine whether a particular segment of a schlieren
    %image exhibits signs of turbulence
    %to do this, essentially take a section of the canny- and
    %high-pass-filtered-image and count the number of columns that have
    %turbulent edges.  
    %These are reference subtracted images, so the edge of a laminar BL
    %should not show up in the image
    function turb_measure = turbulent_section(section)
        
        %find columns with edges above boundary layer 
        [~, edge_c] = find(section);
        edge_c = unique(edge_c);
        
        %take 50% edge presence as turbulent
        if numel(edge_c) > 0.75*size(section,2);
            turb_measure = 1;
        else
            turb_measure = 0;
        end
        
        
    end




    %music isn't the best way to determine whether a particular section is
    %turbulent, so let's use another method (minimum variance?)
    function turb_measure = turbulent_section2(sig_in, M)
        
        %get the average autocorrelation matrix for the high-pass filtered
        %data to assess the presence of turbulence
        Rx = zeros( M+1, M+1);
        for j = 40:40
            [~,Rx_curr] = corrmtx(sig_in(j,:), M);
            Rx = Rx + Rx_curr;
        end
        Rx = Rx/size(sig_in ,1);
        Rx_inv = inv(Rx);

        
        %now find the traces and compute the fft
        q = zeros(1,size(sig_in,1));
        for j = 2:size(Rx_inv,1)
           new_mat = Rx_inv(1:end-j+1,j:end);
           q(j) = trace(new_mat);
        end
        inv_ft = trace(Rx_inv) + 2*real(fft(q, size(sig_in, 2)));
        spec_out = 2*(M+1)./inv_ft;
        
        
        turb_measure = 1;
    end



    %function that crops an image to avoid the penumbra transition region
    function crop_col = crop_shadow(img_in)
        img_in = clamp(img_in(1:cone-5,:));
        mean_edge = mean(clamp(sobel(img_in)), 1);
        smooth_diff = smooth(diff(mean_edge), 35);
        
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


        


end

