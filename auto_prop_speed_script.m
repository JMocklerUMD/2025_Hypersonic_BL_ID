clear all; close all; clc;
% Main script to run the automated finding propogation speed function

% shot number
shot = 33;
img_no = 1001:1010;
no_ref = 100; %15000 --> directory changed in readImagesLangley.m, parfiles, prop_speed_function.m

% length scale [mm/pix]
if sum(33:39 == shot) > 0
    mm_pix = .0756;
elseif sum(79 == shot) > 0
    mm_pix = .0747;
end

% Addpath of necessary directories
% need music_identification_v4.m and all associated codes
% need prop_speed_function.m and filter_notch.m
addpath('J:\wave_packet')
addpath('J:\wave_packet\parfiles_Langley')

duration = length(img_no); 
% img_no will be the variable in the 3rd column of the .mat file outputed by the file you load in

for II = 1:duration
img0 = img_no(II)

[second_mode, packets, BL] = music_identification_v4(shot,img_no(II),no_ref);

%Using the strong packets columns for prop speed...
%if multiple strong packets, just take first packet

if second_mode(2)==1
    if sum(79 == shot) > 0
        col1 = packets{2}(end,1); %run 79 picks up extraneous packets 
        col2 = packets{2}(end,2);
    else
        col1 = packets{2}(1,1);
        col2 = packets{2}(1,2);
    end
    BL;
    row1 = BL(2);
    row2 = BL(1)-5; %-5 to give slightly buffer from cone edge
    cone_height = BL(1);

    if exist('strong_packets','var') == 0
        strong_packets = 1;
        i=1;
    else
        strong_packets = strong_packets+1;
    end
   

    corr_speed(i) = prop_speed_function(shot,img0,col1,col2,row1,row2,mm_pix,cone_height);
    i=i+1;
end
end

corr_speed = corr_speed';

%%
clearvars -except corr_speed strong_packets BL
% Eliminate any speeds that show up as greater than the freestream
% velocity. This happens sometimes because of the correlation, but it's
% easy to eliminate because the speed is much greater than it should be 

corr_speed = corr_speed';
ind = find(corr_speed > 1500);
corr_speed(ind(1:length(ind))) = [];
avg_speed = mean(corr_speed);
stdev_speed = std(corr_speed);

%print to screen
M = ['Number of Usable Speeds = ', num2str(length(corr_speed)), ' out of ', num2str(strong_packets)];
disp(M)
X = ['Average Propagation speed = ',num2str(avg_speed),' m/s']; 
disp(X)
Y = ['Standard Deviation = ',num2str(stdev_speed),' m/s'];
disp(Y)

