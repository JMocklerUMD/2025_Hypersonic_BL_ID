function [turbulenceInfo,turbulent1] = extractInfo(loopRange,shot,no_ref,furthest_image,percentTurbulent,base_dir)
% [turbulenceInfo,turbulent1] = extractInfo(loopRange,shot,no_ref,furthest_image,percentTurbulent)
%
% Inputs - 
%       loopRange - the images to gather information over
%       shot - which image sequence these images are in
%       no_ref - the number of images on either side to include in the
%           refference (averaged) image
%       furthest_image - the last avaliable image in the sequence
%       percentTurbulent - the fraction of the images indicated in
%           loopRange that are expected to have turbulence in them
%
% Outputs - 
%       turbulenceInfo - the input for frequencyIdentification used for 
%           detecting turbulent spots (see documentation for
%           filteredImageIdentification)
%       turbulent1 - the image numbers of images that are expected to have
%           turbulence in them.
%
%
% This function gathers data about the entire image sequence as well as
% identifying images that are likely to be have significant turbulent spots
% in them.



% saved information that I want to look at
% savedInfo = zeros(length(loopRange),3);

% get the correct function to read the images with
imgReadFxn = findImageReadFxn(shot);

% number of points for the fourier transform
pts = 2^16;


% initialize variables to store turbulence info
FTIntegral = zeros(1,length(loopRange));
peakAvgT = zeros(1,length(loopRange));
FTmaxes = FTIntegral;
rMaxes = FTmaxes;

% this is set up to run in parallel to make it faster
% parpool
% parfor loopCount = 1:length(loopRange)

%CANNOT PARALLELIZE THIS DUE TO EVAL STATEMENTS
for loopCount = 1:length(loopRange)
    
    [shot, start, stop, re_ft, res, vert_reflect, horz_reflect,...
        reference_sub, gamma_adjust, pre_rot_row_crop, pre_rot_col_crop, ...
        rotate, fixed_rotation, post_rot_row_crop, post_rot_col_crop, ...
        auto_cone_crop, auto_rot_crop, auto_shadow_crop, bl_method, bl_adjust, bl_invert, peak_range,...
        power_thresh, turb_method, turb_thresh, img_dir, file_format, file_name] = read_par(num2str(shot));
    
    img_num = loopRange(loopCount);
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    img_no = loopRange(loopCount);
    % create the averaged image
    [IM,~] = referenceImage(no_ref,furthest_image,img_no,base_dir,shot,0,0,0);

    fprintf('%d/%d\n', img_no, loopRange(end));

    % read in image file, rotate
    [Img,BLmethod,rotate] = imgReadFxn(base_dir,shot,img_no);
    
    Img = double(Img);
    % Subtract the reference image to get the image that you want to do a
    % fourier transform on
    image_ft =(Img-IM); 
    rotate = 0;
    
    % if the images aren't already properly oriented rotate them
    if rotate
        % Img = coneAdjuster(Img);
        [IM,yp,rowInds,colInds] = coneAdjuster(IM);
        % image_ft = coneAdjuster(image_ft);
        %temp = fit((1:length(yp))',yp','poly1');
        %ang = atan(temp.p1);
        ang = 0;
        Img = imrotate(Img,ang*180/pi,'bilinear');
        image_ft = imrotate(image_ft,ang*180/pi,'bilinear');
        Img = Img(rowInds,colInds);
        image_ft = image_ft(rowInds,colInds);
        tempRows = ceil(yp(end)):size(IM,1);
%         [cone,BL_height,BL_thickness] = findBL(IM(tempRows,:),BLmethod);
%         cone = cone+ceil(yp(end))-1;
%         BL_height = BL_height+ceil(yp(end))-1;
        imgLength = length(image_ft);
        if length(IM) ~= imgLength
            disp(['Something is wrong with image ' num2str(img_no)])
        end
    else
%         [cone,BL_height,BL_thickness] = findBL(IM,BLmethod); % find information about the boundary layer
    end

   
    
    % Fourier Transform

    % run the fourier transform across each row
    FT = fft(image_ft(BL_height:cone,:),pts,2);
    % get the power spectrum by getting |FT|^2
    FT = FT.*conj(FT)/pts;
    % the frequencies of interest
    f = (0:pts/2)/pts;
    
    % frequencies used for turbulence identification
    f_minh = find(f<1/(BL_thickness),1,'last');
    f_maxh = find(f>4/BL_thickness,1,'first');
    
    % information about turbulence (only a little bit of this is currently
    % used)
    [~, ~,peakAvgT(loopCount),~] = turbulenceMeasures(FT,BL_height,cone,Img);
    FTmaxes(loopCount) = max(max(sqrt(FT(:,f_minh:f_maxh))));
%     rMaxes(loopCount) = max(max(sqrt(FT(:,f_min2:f_max2))))/FTmaxes(loopCount);


% filtered information
    
    filtered_img2 = zeros(size(image_ft(BL_height:cone,:)));
    % band pass filter ~ second mode
    bp_filter = fdesign.bandpass(1/(4.5*BL_thickness),(2)/(4.5*BL_thickness),...
        1/(1.5*BL_thickness),(2)/(1.5*BL_thickness),10,2,10);
    bp_filter = design(bp_filter,'cheby1');
    
    for i = 1:length(filtered_img2)
        filtered_img2(:,i) = filter(bp_filter,image_ft(BL_height:cone,i));
    end
    
end

temp = peakAvgT;
turbulent1 = [];
i = 1;
while size(turbulent1,2)<percentTurbulent*length(temp)
    [m,turbulent1(i)] = max(temp);
    temp(turbulent1(i)) = temp(turbulent1(i))-m;
    i = i+1;
end
turbulent1 = sort(turbulent1)+loopRange(1)-1;
% delete(gcp)
turbulenceInfo = FTmaxes;