function [indsNearest,diff] = findNearest(varargin)
%% 
% [indsNearest,diff] = findNearest(locs)
% [indsNearest,diff] = findNearest(locs1,locs2)
%
% Inputs:
%       locs - a unsorted list of numbers
%       --OR--
%       locs1 - a list of numbers where each one will be paired with a
%           element in locs2
%       locs2 - a sorted list of the numbers to be paired with locs1 or a
%           sorted matrix where each row is sorted.
%
% Outputs:
%       indsNearest - the index in locs or locs2 of the nearest number
%       diff - the difference between the nearest number and the origional
%           number
%
%
% Assumes both lists are sorted lowest to highest and then finds the
% indicies of the locations in locs2 that are closest to each element of
% locs1. size(indsNearest) == size(locs1). The value is 0 if locs2 is
% empty.


if size(varargin,2)==1
    [indsNearest,diff] = findNearest1(varargin{1});
elseif size(varargin{2},1)==1
    [indsNearest,diff] = findNearest2(varargin{1},varargin{2});
else
    [indsNearest,diff] = findNearest2D(varargin{1},varargin{2});
end

end

function [indsNearest,diff] = findNearest1(locs)

if max(size(locs)) == 1
    indsNearest = [];
    diff = [];
    return
end

indsNearest = zeros(size(locs));
for i = 1:length(locs)
    temp = abs(locs-locs(i));
    if sum(temp==0>1)
        tinds = find(temp==0);
        tinds = tinds(tinds~=i);
        indsNearest(i) = tinds(1);
    else
        [~,tind] = min(temp(temp~=temp(i)));
        if tind>=i
            tind=tind+1;
        end
        indsNearest(i) = tind;
    end
end

diff = locs-locs(indsNearest);

end

function [indsNearest,diff] = findNearest2(locs1,locs2)

indsNearest = zeros(size(locs1));

for i = 1:length(locs1)
    temp = locs2>locs1(i);
    if sum(temp) > 0
        tind = find(temp,1,'first');
        if tind > 1
            if abs(locs1(i)-locs2(tind))>abs(locs1(i)-locs2(tind-1))
                tind = tind-1;
            end
        end
        indsNearest(i) = (tind);
        
    elseif sum(locs2<=locs1(i)) > 0
        temp = locs2<=locs1(i);
        tind = find(temp,1,'last');
        if tind<length(locs2)
            if abs(locs1(i)-locs2(tind))>abs(locs1(i)-locs2(tind+1))
                tind = tind+1;
            end
        end
        indsNearest(i) = (tind);
        
    end
    
end

diff = locs2(indsNearest) - locs1;

end

function [indsNearest,diff] = findNearest2D(locs1,locs2)

indsNearest = zeros(size(locs1));

for i = 1:length(locs1)
    temp = abs(locs2-locs1(i));
    indsNearest(i) = find(temp == min(min(temp)),1);
    
end

diff = locs2(indsNearest)-locs1;

end