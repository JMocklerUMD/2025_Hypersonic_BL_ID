function [starts,stops] = findBounds(wave_packets,weighting,BL_thickness)
%% function [starts,stops] = findBounds(wave_packets,weighting,BL_thickness)
%
%   Input:
%       wave_packets - The locations from findWavePackets estimating where
%           the boundaries of the wave packets are
%       weighting - the weight that is given to bounds in that row, this
%           should be a number between 0 and 1
%       BL_thickness - the thickness of the bounary layer in pixels
%
%   Outputs:
%       starts - the leftmost locations taken from wave_packets(:,:,1)
%       stops - the rightmost locations taken from wave_packets(:,:,2)
%
%   This function finds the probably starting and ending locations for the
%   2nd mode wave packets that have been identified by wave_packets. Using
%   a weighted voting procedure the beginnings and endings are selected
%   where at least a certain number of the wave_packets agree that the spot
%   begins or ends.
%

% seperate out the beginnings and endings and initialize storage variables
beginnings = (wave_packets(:,:,1));
endings = (wave_packets(:,:,2));
avgRange = 6*BL_thickness; % the allowable difference between a point and the mean
taken = ones(size(beginnings));
taken2 = taken;
starts = [];
stops = [];
% setup to look through beginnings starting with the one that is closest to
% average
% literally impossible to parse this... goddamnit (TJW)
nextPlace = find(abs(taken.*beginnings-mean(mean(beginnings((taken-(beginnings==0))==1))))...
    == min(min(abs(taken.*beginnings-mean(mean(beginnings((taken-(beginnings==0))==1)))))));
Bmean = beginnings(nextPlace(1));
notDone = 1;
% find the averages for the beginnings
while notDone
    % find the ones that could be in range of the mean
    if Bmean > 2*avgRange
        bInRange = find(abs(beginnings.*(taken)-Bmean)<2*avgRange);
    else
        bInRange = find((beginnings<Bmean+2*avgRange).*(taken-(beginnings==0)));
    end
    % get the appropriate wieghtings for all of the possible ones in range
    weightingInRange = weighting(mod(bInRange,length(weighting))+...
        (mod(bInRange,length(weighting))==0)*length(weighting));
    
    % recalculate the mean for the group with the extended range and then
    % narrow it down to the acceptable range
    tempMean = weightedAverage(beginnings(bInRange),weightingInRange);
    % find the ones that could be in range of the mean
    if tempMean > avgRange
        bInRange = find(abs(beginnings.*(taken)-tempMean)<avgRange);
    else
        bInRange = find((beginnings<tempMean+avgRange).*(taken-(beginnings==0)));
    end
    % get the appropriate wieghtings for all of the possible ones in range
    weightingInRange = weighting(mod(bInRange,length(weighting))+...
        (mod(bInRange,length(weighting))==0)*length(weighting));
    
    
    if max(size(bInRange)) == 1
        % if there's no one else
        taken(bInRange) = 0;
        if sum(sum(taken.*beginnings)) ==0
            notDone = ~notDone; % finished
        else
            nextPlace = find(abs(taken.*beginnings-mean(mean(beginnings((taken-(beginnings==0))==1))))...
                == min(min(abs(taken.*beginnings-mean(mean(beginnings((taken-(beginnings==0))==1)))))));
            Bmean = beginnings(nextPlace(1));
        end
    elseif Bmean == weightedAverage(beginnings(bInRange),weightingInRange)
        % done
        taken(bInRange) = 0;
        % set up to take weighting into account
        rows = zeros(length(weighting),1);
        for i = 1:length(rows)-1
            rows(i) = sum(mod(bInRange,length(weighting)) == i);
        end
        rows(end) = sum(mod(bInRange,length(weighting))) == 0;
        % determine if this bound is worth keeping
        if sum(rows+rows.*weighting) > 6
            starts = [starts, mean(beginnings(bInRange))];
        end
        if sum(sum(taken.*beginnings)) ==0
            notDone = ~notDone;
        else
            nextPlace = find(abs(taken.*beginnings-mean(mean(beginnings((taken-(beginnings==0))==1))))...
                == min(min(abs(taken.*beginnings-mean(mean(beginnings((taken-(beginnings==0))==1)))))));
            Bmean = beginnings(nextPlace(1));
        end
    else
        % there are other people to add
        Bmean = weightedAverage(beginnings(bInRange),weightingInRange);
    end
end
% repeat the process for the ends
nextIndexes = find(abs(taken2.*endings-mean(mean(endings((taken2-(endings==0))==1))))...
    == min(min(abs(taken2.*endings-mean(mean(endings((taken2-(endings==0))==1)))))));
Emean = endings(nextIndexes(1));
notDone = 1;
while notDone
    if Emean > 2*avgRange
        eInRange = find(abs(endings.*(taken2)-Emean)<2*avgRange);
    else
        eInRange = find((endings<Emean+2*avgRange).*(taken2-(endings==0)));
    end
    % get the appropriate wieghtings for all of the possible ones in range
    weightingInRange = weighting(mod(eInRange,length(weighting))+...
        (mod(eInRange,length(weighting))==0)*length(weighting));
    
    % recalculate the mean for the group with the extended range and then
    % narrow it down to the acceptable range
    tempMean = weightedAverage(endings(eInRange),weightingInRange);
    % find the ones that could be in range of the mean
    if tempMean > avgRange
        eInRange = find(abs(endings.*(taken2)-tempMean)<avgRange);
    else
        eInRange = find((endings<tempMean+avgRange).*(taken2-(endings==0)));
    end
    % get the appropriate wieghtings for all of the possible ones in range
    weightingInRange = weighting(mod(eInRange,length(weighting))+...
        (mod(eInRange,length(weighting))==0)*length(weighting));
    
    
    if max(size(eInRange)) == 1
        % if there's no one else
        taken2(eInRange) = 0;
        if sum(sum(taken2.*endings)) ==0
            notDone = ~notDone;
        else
            nextPlace = find(abs(taken2.*endings-mean(mean(endings((taken2-(endings==0))==1))))...
                == min(min(abs(taken2.*endings-mean(mean(endings((taken2-(endings==0))==1)))))));
            Emean = endings(nextPlace(1));
        end
    elseif Emean == weightedAverage(endings(eInRange),weightingInRange);
        % done
        taken2(eInRange) = 0;
        % set up to take weighting into account
        rows = zeros(length(weighting),1);
        for i = 1:length(rows)-1
            rows(i) = sum(mod(eInRange,length(weighting)) == i);
        end
        rows(end) = sum(mod(eInRange,length(weighting))) == 0;
        % determine if this bound is worth keeping
        if sum(rows+rows.*weighting) > 6
            stops = [stops, mean(endings(eInRange))];
        end
        if sum(sum(taken2.*endings)) ==0
            notDone = ~notDone;
        else
            nextPlace = find(abs(taken2.*endings-mean(mean(endings((taken2-(endings==0))==1))))...
                == min(min(abs(taken2.*endings-mean(mean(endings((taken2-(endings==0))==1)))))));
            Emean = weightedAverage(endings(eInRange),weightingInRange);
        end
    else
        % there are other people to add
        Emean = weightedAverage(endings(eInRange),weightingInRange);
    end
end

end