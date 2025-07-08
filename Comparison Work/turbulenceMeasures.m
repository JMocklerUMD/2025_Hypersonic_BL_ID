function [FTintegral,nPeaks,peakAvg,peakAvg2] = turbulenceMeasures(FT,BL_height,cone,IM)
%%[FTintegral,nPeaks,peakAvg,cannyResult] = turbulenceMeasures(FT,BL_height,cone,IM)
%
%   Inputs:
%       FT - the frequency information from the fourier transform. This
%           covers the entire boundary layer
%       BL_height - the top of the boundary layer in IM
%       cone - the location of the cone in IM
%       IM - the image to be used for canny edge detection
%
%   Outputs:
%       FTintegral - the integral of the frequency space in the range
%           between find_min and find_max
%       nPeaks - the number of peaks in the same interval
%       peakAvg - the average height of a peak in the 1st harmonic range
%       peakAvg2 - the average peak height in the 2nd mode range
%
% This function calculates various information related to turbulence. It is
% normally used in preprocessing of an image sequence.
%

%% frequency information

% find the frequency bounds to use (centered on 1st harmonic)
BL_thickness = cone-BL_height;
f = (0:2^16/2)/2^16;
find_min = find(f>1/(2*BL_thickness),1,'first');
find_max = find(f<(1/(BL_thickness*1)),1,'last');
% Frequency bounds for 2nd mode frequency
find_min2 = find(f>1/(4.5*BL_thickness),1,'first');
find_max2 = find(f<(1/(BL_thickness*1.5)),1,'last');

% Integrate the whole frequency spectrum, then devide by the "area" to
% determine the approximate average power value for the frequency range of
% interest
BL_thickness = cone-BL_height;
FTintegral = sum(sum((FT(1:end-1,find_min:find_max-1)+...
    FT(1:end-1,find_min+1:find_max)+FT(2:end,find_min:find_max-1)+...
    FT(2:end,find_min+1:find_max))/4))/(BL_thickness*(find_max-find_min));

% finding peaks in the frequency spectrum in each row to create two
% additional measures to determine if the image has turbulence
peakTotal = 0;
peakTotal2 = 0;
nPeaks = 0;
nPeaks2 = 0;
threshold = 0;% 10^floor(log10(max(max(FT(:,find_min:find_max)))/3));
for i = 1:size(FT,1)
    peaks = findpeaks(FT(i,find_min:find_max));
    peaks = peaks(peaks>threshold);
    % find the number of peaks and the average height
    nPeaks = nPeaks+numel(peaks);
    for peak = peaks
        peakTotal=peakTotal+(peak);
    end
    peaks = findpeaks(FT(i,find_min2:find_max2));
    peaks = peaks(peaks>threshold);
    nPeaks2 = nPeaks2+numel(peaks);
    for peak = peaks
        peakTotal2=peakTotal2+(peak);
    end
end
% average out the peaks
peakAvg = peakTotal/nPeaks;
peakAvg2 = peakTotal2/nPeaks2;

%% the canny edge detection
% smear out the cone and increase the contrast in the image. In order to
% make sure that the cone is completely smeared out the next two rows up
% are included as well. Also when increasing the contrast a gamma value of
% 5 worked well in testing.
IM = (IM-min(min(IM)))/(max(max(IM))-min(min(IM)));
IM(end:-1:cone-2,:) = mean(IM(cone-3,:));
IM = imadjust(IM,[0; 1],[0; 1],5);
end