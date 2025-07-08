function [Pairs] = pairBounds(starts,stops,wave_packets,BL_thickness,weighting)
%% [Pairs] = pairBounds(starts,stops,wave_packets,BL_thickness)
%
%   Inputs:
%       starts - the locations where wave packets are assumed to start, aka
%           the left boundaries
%       stops - the locations where the wave packets are assumed to stop,
%           aka the right boundaries
%       wave_packets - the output from findWavePackets
%       BL_thickness - the thickness of the boundary layer calculated from
%           the Schlieren image
%       weighting - the weights for each row based on the fourier transform
%
%   Output:
%       Pairs - pairings of elements of starts and stops that bracket wave
%           packets
%
% This function is used to determine pairs of bounds. Using the information
% from wave_packets the starts and stops first based on whether they are
% near locations indicated in the same rows across the image. If there
% aren't any sutible matches this way then they are simply paired with the
% closest one. After all of the starts and stops are paired calculations
% are done to determine which ones will stay based on size and overlap. The
% final results are returned in pairs which contains the indexes for starts
% (Pairs(:,1)) and stops (Pairs(:,2)) that give the paired boundaries.
%

% initialize variables to be used later
startSupport = {};
stopSupport = {};
beginnings = wave_packets(:,:,1);
endings = wave_packets(:,:,2);
Pairs = [];

% Deternime which rows have beginnings that are in the range of this start
for i = 1:length(starts)
    [startSupport{i},~] = find((beginnings>starts(i)-100).*...
        (beginnings<starts(i)+100));
end

% repeat for the endings
for i = 1:length(stops)
    [stopSupport{i},~] = find((endings>stops(i)-100).*(endings<stops(i)+100));
end
% the use of a range of +-100 is somewhat arbitrary, but appears to work
% reasonably well.


% If there is only one pair keep it.
if length(starts(starts>0)) == 1 && length(stops(stops>0)) == 1
    [Pairs(1,1),~] = find(starts>0);
    [Pairs(1,2),~] = find(stops>0);
    
else
    % create pairs of bounds
    % identify initial pairs by just pairing the closest ones
    StartPair = zeros(length(starts),2);
    for i = 1:length(starts)
        % identify initial pairs by just pairing the closest ones or the ones
        % that are the best match based on information from wave_packets
        pairedStop = 0;
        agreement = 0;
        
        % find which stop has the most similar support
        for j = 1:length(stopSupport)
            tempAgreement = intersect(startSupport{i},stopSupport{j});
            if length(tempAgreement) > agreement
                agreement = length(tempAgreement);
                pairedStop = j;
            end
        end
        
        % set tempInd
        if pairedStop ~= 0
            tempInd = pairedStop;
        else
            % determine the distance between all of the stops and the current start
            diff = stops-starts(i);
            % make sure that all of the differences are positive
            diff = diff.*(diff>5*BL_thickness)+1000*(diff<=5*BL_thickness);
            % find the smallest difference
            tempInd = find((diff == min(diff)).*(diff<1000));
        end
        
        % check to make sure there is a pair
        if ~isempty(tempInd)
            % check the validity of the pair
            [Trows,~] = find((beginnings>starts(i)-100).*...
                (beginnings<starts(i)+100));
            [Prows,~] = find((endings>stops(tempInd)-100).*...
                (endings<stops(tempInd)+100));
            same = intersect(Trows,Prows);
            if length(same)> 0.3*min(length(Trows),length(Prows))
                StartPair(i,:) = [i,tempInd];
            end
        end
    end
    
    % repeat going through stops
    StopPair = zeros(length(stops),2);
    for i = 1:length(stops)
        diff = stops(i)-starts;
        diff = diff.*(diff>5*BL_thickness)+1000*(diff<=5*BL_thickness);
        tempInd =find((diff == min(diff)).*(diff<1000));
        if ~isempty(tempInd)
            % check the validity of the pair
            [Trows,~] = find((beginnings>starts(tempInd)-100).*...
                (beginnings<starts(tempInd)+100));
            [Prows,~] = find((endings>stops(i)-100).*...
                (endings<stops(i)+100));
            same = intersect(Trows,Prows);
            if (length(same)> 0.3*max(length(Trows),length(Prows))) || ...
                    abs(length(Trows)-length(Prows)) < 0.3*min(length(Trows),length(Prows))
                StopPair(i,:) = [tempInd,i];
            end
        end
    end
    % now change that so that the pairs are in places that make sense (ok, so
    % hopefully they'll make sense)
    for i = 1:size(StartPair,1)
        % check to see if this start shares a stop with any other starts
        % and keep the largest one (the one furthest along the image)
        if StartPair(i,1) ~= 0
            diff = stops(StartPair(i,2))-starts(StartPair(i,1));
            rows = find(StartPair(:,2)==StartPair(i,2));
            for j = rows'
                if diff < stops(StartPair(j,2))-starts(StartPair(j,1))
                    StartPair(i,:) = 0;
                    break
                end
            end
            % check to see if there is a StopPair that includes the stop
            % from the current pair and keep the larger pair
            if StartPair(i,2) ~= 0
                rows = find(StopPair(:,2) == StartPair(i,2));
                for j = rows'
                    if diff < stops(StopPair(j,2))-starts(StopPair(j,1))
                        StartPair(i,:) = 0;
                        break
                    end
                end
            end
            % if this pair encloses another pair, get rid of the other pair
            if StartPair(i,1) ~= 0
                for j = 1:max(size(StartPair(:,1)))
                    if StartPair(j,1) ~= 0
                        if starts(StartPair(j,1))>starts(StartPair(i,1)) && ...
                                stops(StartPair(j,2))<stops(StartPair(i,2))
                            StartPair(j,:) = 0;
                        end
                    end
                end
            end
        end
    end
    for i = 1:size(StopPair,1)
        if StopPair(i,1) ~= 0
            diff = stops(StopPair(i,2))-starts(StopPair(i,1));
            rows = find(StopPair(:,1)==StopPair(i,1));
            for j = rows'
                if diff < stops(StopPair(j,2))-starts(StopPair(j,1))
                    StopPair(i,:) = 0;
                    break
                end
            end
        end
        if StopPair(i,1) ~= 0
            rows = find(StartPair(:,1) == StopPair(i,1));
            if isempty(rows)
                % get the index of any boundaries that are in 
                % between the current start-stop pair
                left = find((stops<stops(StopPair(i,2))).*...
                    (stops>starts(StopPair(i,1))));
                % if there aren't any and this pair doesn't exist 
                % in the startpair array then it's tossed out
                if isempty(left)
                    StopPair(i,:) = 0;
                else
                    % if there are some check to see if they are in the
                    % startpair array and if so then create a bigger
                    % boundary and eliminate the one in startpairs
                    if isrow(left)
                        left = left';
                    end
                    changed = 0;
                    for k = left'
                        tempInd = find(StartPair(:,2)==k);
                        if ~isempty(tempInd)
                            StopPair(i,1) = StartPair(tempInd,1);
                            StartPair(tempInd,:) = 0;
                            changed = changed+1;
                        else
                            if find(left == k) == max(size(left)) && changed ==0
                                StopPair(i,:) = 0;
                            end
                        end
                    end
                end
            else
                % if there are other pairs with the same start keep the
                % largest one
                for j = rows'
                    if diff > stops(StartPair(j,2))-starts(StartPair(j,1))
                        StartPair(j,:) = 0;
                    else
                        StopPair(i,:) = 0;
                    end
                end
            end
        end
    end
    Pairs = [StartPair(StartPair(:,1)~=0,:) ; StopPair(StopPair(:,1)~=0,:)];
end

end