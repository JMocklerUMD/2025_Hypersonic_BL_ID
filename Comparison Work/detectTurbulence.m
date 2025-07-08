function [spots,turbulentMeasure] = detectTurbulence(filtered_img1,FT,turbulenceInfo,f_min,f_max)
% [spots,turbulentMeasure] = detectTurbulence(filtered_img1,FT,turbulenceInfo,f_min,f_max)
%
%   Inputs:
%       filtered_img1 - the image filtered to retain high frequency
%           information
%       FT - the fourier transform power for the rows in the boundary layer
%       turbulenceInfo - the maximum FT value around the 1st harmonic for
%           each image in the sequence
%       f_min - the index in FT where the approximate range of the first
%           harmonic starts
%       f_max - the index in FT where the approximate range of the first
%           harmonic ends
%   Outputs:
%       spots - A Nx2 matrix that contains the starting locations (1st
%           column) and ending locations (2nd column) of the identified
%           turbulent spots. If there are no identified turbulent spots
%           this is [0,0]
%       turbulentMeasure - The measure of turbulence used to detect
%           turbulent spots
%
% This function identifieds turbulent spots based on high frequency
% information from the image.
%

% threshold adjustments
averageMag = 3;
stepUp = 0.2;
fractionZero = 0.2;
maxAbove = 10;

% combine information from the particular image and image sequence to
% determine the proper turbulence threshold
fMax = max(max(sqrt(FT(:,f_min:f_max))));
fMean = mean(turbulenceInfo);
fSTD = std(turbulenceInfo);
thresholdT = fMean+(2+(fMax-fMean)/(fSTD*3))*fSTD;

% testing
if max(size(thresholdT))>1
    if max(size(fMax))>1
        disp('fMax problem');
    elseif max(size(fMean))>1
        disp('fMean problem');
    elseif max(size(fSTD))>1
        disp('fSTD problem');
    end
end

%     testing finding turbulent spots
thresholdTest = abs(filtered_img1).*(abs(filtered_img1)>thresholdT);
turbulentMeasure = sum(thresholdTest./thresholdT,1);
spots = [0,0];
tind = 1;
currentSpot = [0 0];


for i = 1:length(turbulentMeasure)
    if i>currentSpot(1) && i<=currentSpot(2)
%             if you're in the current spot then don't do anything
    else
        j = 0;
        spotMax = 0; % maximum average in the spot
        spotMaxInd = 0; % index of maximum average in the spot
        descendingShift = 0; % change in percentage allowed to be 0
%       determine if it is a turbulent spot
        while (mean(turbulentMeasure(i:i+j)) > 1+j*stepUp || ...
            mean(turbulentMeasure(i:i+j)) > averageMag) && ...
            sum(turbulentMeasure(i:i+j)==0)/(j+1) < fractionZero-descendingShift && ...
            i+j<length(turbulentMeasure) && ...
            maxAbove < max(turbulentMeasure(i:i+j))
            j = j+1;
            if mean(turbulentMeasure(i:i+j)) > spotMax
                spotMax = mean(turbulentMeasure(i:i+j));
                spotMaxInd = i+j;
            end
            descendingShift = (i+j-spotMaxInd)*0.0006;
        end
%             check to be sure that it's long enough
        if (j>40 || (j>20 && i+j == length(turbulentMeasure)))
            currentSpot = [i i+j];
            spots(tind,:) = currentSpot;
            tind = tind+1;
        end
    end

end



end