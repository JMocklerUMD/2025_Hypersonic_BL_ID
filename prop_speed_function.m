function [corr_speed] = prop_speed_function(shot,img0,col1,col2,row1,row2,mm_pix,cone_height)

% this function is an adaptation of the propogation speed script so that I
% can read in the information from Tom's codes. % calculates the correlation coefficients betwee a reference and displaced
% images over a specified region to determine the velocity

global angle

image_moved = 1;
img1 = img0+image_moved; % IMG0 manually entered based on knowledge of where wave packet starts

% make sure the rows Tom gave are measured from the bottom up
row = row1:row2;
no_ref = 100; %number of reference images to average over
ref_range = img0-no_ref:img1+no_ref; % reference range
row_filt = [];

% parameters for correlation
min_disp = 30;
min_disp = (img1-img0)*min_disp;

disp_mult = 50; % maximum number of pixels difference per image to consider 
max_disp = (img1-img0)*disp_mult;
disp_range = min_disp:max_disp;

if col2 > (1150-max_disp) %Gauruntee that the 2nd column of displaced
    col2=1150-max_disp; %image does not exceed image end column
end
col = col1:col2;

ww = 3; % windowing parameter

if sum(33:39 == shot) > 0
    fr = 285.70;
    if col2 > (1150-max_disp)
        col2=1150-max_disp;
    end
    col = col1:col2;
elseif sum(79 == shot) > 0
    fr = 450;
    if col2 > (850-max_disp)
        col2=850-max_disp;
    end
    col = col1:col2;
end

dt = 1/fr; 

dx = (min_disp:max_disp)*mm_pix/1000; % dx = physical distance to move left to right

% Now find the rotation information 
img_no = [img0,img1];
points = 2^16;

for ii = 1:2
    im_dir = sprintf('J:\\NASA Langley Cone Experiments\\Run%d\\Run %d TIFFs 15000',shot,shot);    
    ref_range = img_no(ii)-no_ref:img_no(ii)+no_ref; % reference range
    nn=0;
    for KK = 1:length(ref_range)
        im_file = sprintf('%s\\run%d.%06d.tif',im_dir,shot,ref_range(KK));
        I = imread(im_file);
        if KK == 1
            IM = double(I);
        else
            IM = IM + double(I);
        end
        nn = nn + 1;
    end
    IM = IM/nn;
    
    im_file = sprintf('%s\\Run%d.%06d.tif',im_dir,shot,img_no(ii)); %changed directory **RK
    I = imread(im_file);
    angle = 0; % angle of cone in image
    In = double(I) - IM; 
    In = imrotate(In,angle*180/pi);
    
    if sum(33:39 == shot) > 0
        In = In(1:cone_height,1:1150);
    elseif sum(79 == shot) > 0
        In = In(1:cone_height,1:850);
    end
    
    
    % apply band-pass filter:
    
    % parameters for notch filter:
    % find approximate bounds of the wavelength (in pixels) of your
    % dominant disturbance... [low_bound_pixels , high_bound_pixels]
    % set cut_low = 1/low_bound_pixels
    % set cut_high = 1/high_bound_pixels
    
    cut_low = 1/35; cut_high = 1/45;

    order_low = 6; order_high = 6;
    
    for jj = 1:length(row)
        vt = In(row(jj),:)';
        Itc = filter_notch(vt,...
            cut_low,cut_high,order_low,order_high)';
        
        In(row(jj),:) = Itc;
    end
    
    if ii == 1
        vi = In(row,col);
        vref = vi;
    else
        vdisp = In;
  end
  
end

% calculate correlation coefficients for different displacements
for jj = 1:length(disp_range)
  vdispt = vdisp(row,col+disp_range(jj)); %
  cc(jj) = corr2(vref,vdispt); %2D correlation coefficient
end

% find maximum point and fit quadratic to ww points on either side
[max_cc,ind] = max(cc); % maximum correlation coefficient and index
if ind-ww<1
    dx_cut = dx(ind:ind+ww);
    cc_cut = cc(ind:ind+ww);
else if ind+ww>= max(jj)
        dx_cut = dx(ind-ww:ind);
        cc_cut = cc(ind-ww:ind);
    else
        dx_cut = dx(ind-ww:ind+ww); % dx centered around index of max correlation coefficient
        cc_cut = cc(ind-ww:ind+ww); % cc centered around index of max correlation coefficient
    end
end



dx_interp = linspace(min(dx_cut),max(dx_cut),1000); %x value over which to interpolate
p = polyfit(dx_cut,cc_cut,2);
cc_interp = p(1)*dx_interp.^2 + p(2)*dx_interp + p(3);

% create u vectors and plot results
u = dx/dt*1e3;
u_interp = dx_interp/dt*1e3;

[cc_max,max_ind] = max(cc_interp);
corr_speed = u_interp(max_ind)/image_moved;
