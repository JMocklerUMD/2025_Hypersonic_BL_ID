function [cone,BL_height,BL_thickness] = findBL(Img,method)
%% [cone,BL_height,BL_thickness] = findBL(Img,method)
%
%   Input:
%       Img - the image that the boundary layer is to be found in
%       method - select what method is to be used to find the BL height.
%           1 - the origional method which looks for the heighest point
%               where in each column and takes the median of those results
%           2 - the modified method which takes the median of the gradients
%               above the mean in the area of the image above the cone
%
%   Output:
%       cone - the row where the cone stops intruding into the image
%       BL_height - the highest row where the boundary layer can be clearly
%           identifed
%       BL_thickness - the thickness of the boundary layer
%
%   This function uses edge finding techniques to determine where the
%   largest gradients are to determine where the cone stops and where the
%   boundary layer ends.
%

%% do the edge finding

Edge_mag = zeros(size(Img));


% this uses a sobel filter to calculate the gradient (sobel filters are 1 2
% 1 rather than 1 1 1)
Edge_horz = 1/8*(Img(1:end-2,1:end-2)+2*Img(2:end-1,1:end-2)+Img(3:end,1:end-2)-Img(1:end-2,3:end)-2*Img(2:end-1,3:end)-Img(3:end,3:end));
Edge_vert = 1/8*(-Img(1:end-2,1:end-2)+Img(3:end,1:end-2)-2*Img(1:end-2,2:end-1)+2*Img(3:end,2:end-1)-Img(1:end-2,3:end)+Img(3:end,3:end));
Edge_mag(2:end-1,2:end-1) = (Edge_vert.^2+Edge_horz.^2).^(1/2);


% average across rows to eliminate noise
Edge_mag = [sum(Edge_mag(:,1:3),2)/3,sum(Edge_mag(:,1:4),2)/4,...
    (Edge_mag(:,1:end-4)+Edge_mag(:,2:end-3)+Edge_mag(:,3:end-2)+...
    Edge_mag(:,4:end-1)+Edge_mag(:,5:end))/5,sum(Edge_mag(:,end-3:end),2)/4,...
    sum(Edge_mag(:,end-2:end),2)/3];


% determine the location of the cone and the approximate location of the
% top of the boundary layer
coneThreshold = 8*mean(mean(Edge_mag));
[row,~] = find(Edge_mag>coneThreshold);
% cone = max(row);
cone = mode(row);    %slight modification; use the mode here instead of max
rows = ones(1,length(Edge_mag));    %prevents accidental identification
while numel(cone) == 0 || isnan(cone)
    coneThreshold = 0.9*coneThreshold;
    [row,~] = find(Edge_mag>coneThreshold);
    cone = mode(row);
end
    

% For finding the boundary layer look at each column seperately and take
% the median
% imgHeight = size(Img,1);
% area where the top of the boundary layer should be

% BLzone = 1:cone-6;
BLzone = 1:cone-3;  %changed to account for very thin boundary layers

% 7/24 NOTE This worked for the four origional sequences, but the new
% sequences it doesn't work for so I'm changing the method and I hope it
% works for everyone, nevermind it didn't, now there are two methods
switch method
    case 1
        for i = 1:length(Edge_mag)
            [row,~] = find((Edge_mag(BLzone,i)>mean2(Edge_mag(BLzone,:))));
            if ~isempty(row)
                rows(i) = min(row);
            end
        end
        
    case 2
        % This is the new method
        [rows,~] = find(Edge_mag(BLzone,:)>mean2(Edge_mag(BLzone,:)));

    case 3
        avg_edge = mean(Edge_mag(BLzone, :), 2);
        [~, rows] = findpeaks(smooth(avg_edge,3), 'sortstr','descend', 'npeaks',2);
        if (cone - rows(1)) < 6   %in case we have steep cone-reflection gradient
            rows = rows(2);
        else
            rows = rows(1);
        end
        
    otherwise
        for i = 1:length(Edge_mag)
            [row,~] = find((Edge_mag(BLzone,i)>mean2(Edge_mag(BLzone,:))));
            if ~isempty(row)
                rows(i) = min(row);
            end
        end
end

BL_height = floor(median(rows));
% find the thickness of the boundary layer
BL_thickness = cone-BL_height;


end