function [weightedAvg] = weightedAverage(values,weighting)
%% weightedAvg = weightedAverage(values,weighting)
%
%   Inputs:
%       values - the values to be averaged in a 1 dimensional vector
%       weighting - their corrosponding weightings in an indetically sized
%           1 dimensional vector
%
%   Output:
%       weightedAvg - the weighted average using the weightings given

weightedAvg = sum(values.*weighting/sum(weighting));

end