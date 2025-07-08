function [turbulentSpots] = waveletTurbulence(resultC,BL_thickness,maxSpread,minSpread,thresholdShift)
%% turbulentSpots = waveletTurbulence(resultC,BL_thickness,maxSpread,minSpread)
%
%   Inputs:
%       resultC - the result of the wavelet transform using the modified
%           basis function
%       BL_thickness - the thickness of the boundary layer extimated from
%           the averaged image
%       maxSpread - the largest b value used in the transform
%       minSpread - the smallest b value used in the transform
%       thresholdShift - the adjustment to the threshold used to detect
%           turbulent spots. The base threshold is 0.8 and is modified 
%           before being multiplied by the BL_thickness
%   Output:
%       turbulentSpots: the beginning and ending location of any turbulent
%           spots found. turbulentSpots(1,:) = beginnings;
%           turbulentSpots(2,:) = endings;
%
% This function determines the location and extent of turbulent spots based
% on the wavelet transform information in resultC.
%

% set thresholds
highThreshold = 0.8+thresholdShift;
lowThreshold = 0.4+thresholdShift/2;

tempBW = resultC>highThreshold*(BL_thickness+1)/10;
labeledSpots = bwlabel(tempBW);
% notCovered = ones(1,max(max(labeledSpots)));
turbulentSpots = zeros(1,2);
tind = 1;
for i = 1:max(max(labeledSpots))
    [~,cols] = find(labeledSpots==i); % make sure this spot hasn't been counted
    if sum(min(cols)<turbulentSpots(:,2) & max(cols)>turbulentSpots(:,1)) == 0
        % get the location of the spot
        tempBW = labeledSpots == i;
        peak = max(max(tempBW.*resultC));
        [row,col] = find(tempBW.*resultC==peak,1);
        % use the low threshold to get the full extent
        tempBW = resultC>lowThreshold*(BL_thickness+1)/10;
        tempBW = bwlabel(tempBW);
        tempBW = tempBW == tempBW(row,col);
        [rows,cols] = find(tempBW);
        if max(rows)-min(rows) == maxSpread-minSpread
            lefts = zeros(1,max(rows));
            rights = lefts;
            for j = 1:max(rows)
                lefts(j) = min(cols(rows==j));
                rights(j) = max(cols(rows==j));
            end
            % I could put something here about checking the std of rights
            % and lefts
            turbulentSpots(tind,:) = [max(lefts),max(rights)];
            tind = tind+1;
        end
    end
end


end