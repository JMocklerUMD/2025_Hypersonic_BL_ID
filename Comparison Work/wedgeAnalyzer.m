function [goodWedge,lefts,rights,tempBW] = wedgeAnalyzer(resultA,peak,row,col,BL_thickness,tightRestriction)
%% [goodWedge,lefts,rights,tempBW] = wedgeAnalyzer(resultA,peak,row,col,BL_thickness,tightRestriction)
%
%   Inputs:
%       resultA - the modified wavelet analysis result
%       peak - the maximum value that will be used to find the wedge
%       row - the row location of peak in resultA
%       col - the column location of peak in resultA
%       BL_thickness - the thickness of the boundary layer estimated from
%           the averaged image
%       tightRestriction - a logical value determining how tight the
%           restrictions on a wedge should be
%
%   Outputs:
%       goodWedge - a logical value indicating whether the wedge meets the
%           criterion
%       lefts - the left boundary indicated by the various rows in the
%           wedge
%       rights - the right boundary indicated by the various rows in the
%           wedge
%       tempBW - the binary image of the current wedge
%
% This function analyzes the wedge like shapes that are seen in the
% modified wavelet transform at the location of second mode wave packets.
%


% stuff that should go into a seperate function when this
% is officially finished
dropoffPercentStart = 0.8; % variable used in identifying packets
widthVariation = 0.25; % maximum percentage variation in location/width

for perc = dropoffPercentStart:0.01:0.99
    % isolate the section that is connected to this peak
    tempBW = resultA>perc*peak; % get the values above the threshold
    tempBW = bwlabel(tempBW);
    tempBW = tempBW==tempBW(row,col); % isolate the group connected to the peak

    [rows,cols] = find(tempBW);

    % preallocate space
    widths = zeros(1,max(rows)-min(rows)+1);
    lefts = widths;
    rights = lefts;

    % find the width of the packet indicated by each row
    for k = min(rows):max(rows)
        addedWidth =  k*BL_thickness;
        lefts(k-min(rows)+1) = min(cols(rows==k))-addedWidth/2;
        rights(k-min(rows)+1) = max(cols(rows==k))+addedWidth/2;
        widths(k-min(rows)+1) = max(cols(rows==k))-min(...
            cols(rows==k))+addedWidth;
    end

    % make sure the top is much smaller than the bottom
    topW = max(cols(rows==max(rows)))-min(cols(rows==max(rows)));
    bottomW = max(cols(rows==min(rows)))-min(cols(rows==max(rows)));
    acceptableVariation = min(6*BL_thickness,widthVariation*mean(widths));
    if tightRestriction
        goodWedge = length(widths)>3 && max(abs(lefts-... %MAGIC NUMBER
            mean(lefts)))<acceptableVariation && max(abs(...
            rights-mean(rights)))<acceptableVariation && ...
            topW<0.5*bottomW;
    else
        goodWedge = length(widths)>3 && mean(abs(lefts-... %MAGIC NUMBER
            mean(lefts)))<acceptableVariation && mean(abs(...
            rights-mean(rights)))<acceptableVariation && ...
            topW<0.5*bottomW;
    end
    if goodWedge
        break;
    end
    % end of helper function
end

end