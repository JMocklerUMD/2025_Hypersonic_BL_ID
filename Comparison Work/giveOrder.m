function [rank] = giveOrder(list,highLow)
%% [rank] = giveOrder(list,highLow)
%
%   Inputs:
%       list - the list to rank
%       highLow - give rank from high to low (1) or low to high (0).
%           Treated as a boolian value.
%
%   Outputs:
%       rank - the rank of each element in the list (1-length(list)) with
%           the possibility for two of the same rank. Each rank corrosponds
%           to the element in the same location in list.
%
%

rank = zeros(size(list));

for i = 1:length(list)
    if highLow
        rank(i) = sum(list>list(i))+1;
    else
        rank(i) = sum(list<list(i))+1;
    end
end

end