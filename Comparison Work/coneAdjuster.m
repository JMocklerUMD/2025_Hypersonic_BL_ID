function [Img, varargout] = coneAdjuster(I)
% [Img, [yp, rowInds, colInds, blx]] = coneAdjuster(I)
%
%   Input:
%       I - the image that must be rotated because the model is at an angle
%   Output:
%       Img - the rotated image
%   Optional Outputs:
%       yp - the row component of the location of the cone in I
%       rowInds - the rows of the rotated image that were retained
%       colInds - the columns of the rotated image that were retained
%       blx - the first column where the cone appears in I
%
% This function rotates I so that the model is horizontal then extracts the
% section above the model.
%

% set up variables to look for cone
xmax = length(I); 
ymax = size(I,1);
slopes = zeros(9,1);
intercepts = slopes;
tind =1;

% look along 9 different lines for the cone. Four of them come out of the
% lower right corner at various angles the other five are verticle lines at
% various points across the image
for ang = [80 60 40 20]*pi/180
    if ang > pi/4
        xs = xmax-cos(ang)*(1:ymax);
        ys = ymax:-1:1;
        xs = xs(xs>=1);
        ys = ys(xs>=1);
    else
        ys = ymax-sin(ang)*(1:xmax);
        xs = xmax:-1:1;
        xs = xs(ys>=1);
        ys = ys(ys>=1);
    end
    [slopes(tind),intercepts(tind)] = coneFinder(I,xs,ys);
    tind = tind+1;
end

for pos = (1:5)/6*length(I)
    ys = ymax:-1:1;
    xs = ones(size(ys))*pos;
    [slopes(tind),intercepts(tind)] = coneFinder(I,xs,ys);
    tind = tind+1;
end
% weight the different results differently based on the ones which are
% more likely to have produced the correct cone result
weighting = [1; 2; 2; 1; 0.5; 1; 2; 2; 1];
tind = slopes~=0;
slope = median([slopes(weighting.*tind>0)' slopes(weighting.*tind>=1)' slopes(weighting.*tind>=2)']);%weightedAverage(slopes(slopes~=0),weighting(slopes~=0)');
% intersept = weightedAverage(intersepts(slopes~=0),weighting(slopes~=0)');
[~,tind] = min(abs(slopes-slope));
slope = slopes(tind);
angle = atan(slope);
% [~,tind] = min(abs(intersepts-intersept));
intercept = intercepts(tind);
yp = slope*(1:length(I))+intercept;

if abs(slope) > 1/length(I)
    % adjust image to get the BL horizontal
    Img = imrotate(I,angle*180/pi,'bilinear');
    % start point for the cone
    blx = find(yp<ymax,1,'first');
    tcol = ceil(blx*cos(angle));
    bottomRows = zeros(1,5);
    step = floor((length(Img)-tcol)/(length(bottomRows)));
    for ii = 1:5
        temp = Img(:,tcol+(ii-1)*step);
        trow = find(temp>0,1,'last');
        if ~isempty(trow)
            bottomRows(ii) = trow;
        end
    end
    % cut off black space on the image
    trow = ceil(median(bottomRows(bottomRows~=0)));
    if trow > intercept
        if angle > 0
            trow = ceil(intercept + size(I,2)*sin(angle) + 2);
        else
            trow = ceil(intercept) + 2;
        end
    end
    if trow<size(Img,1)
        Img = Img(1:trow+1,:);
    else
        trow = size(Img,1)-1;
    end
    [~,tcols] = find(Img>0);
    % cut off the right side and the left infront of where the cone
    % intersects the bottom of the image.
    if trow>ymax
        rowInds = trow+1-ymax:trow+1;
    else
        rowInds = 1:size(Img,1);
    end
    colInds = tcol:max(tcols);
    Img = Img(rowInds,colInds);
else
    Img = I;
    rowInds = 1:size(Img,1);
    colInds = 1:size(Img,2);
    blx = 1;
end

switch nargout
    case 2
        varargout = {yp};
    case 3
        varargout = {yp,rowInds};
    case 4
        varargout = {yp,rowInds,colInds};
    case 5
        varargout = {yp,rowInds,colInds,blx};
    otherwise
        varargout = {};
end

% if nargout > 1
%     varargout = {yp};
%     if nargout > 2
%         varargout = {varargout,rowInds};
%         if nargout > 3
%             varargout = {varargout
%         varargout = {varargout,blx};
%     end
% end

% code that shifted the columns down which doesn't properly preserve the
% structure (you should have thought of that earlier!)
% for ii = 1:length(I)
%     ypInd = floor(yp(ii));
%     if ypInd < ymax && ypInd > 0
%         Img(ymax:-1:(ymax-ypInd+1),ii) = I(ypInd:-1:1,ii);
%     else
%         Img(:,ii) = I(:,ii);
%     end
% end

end