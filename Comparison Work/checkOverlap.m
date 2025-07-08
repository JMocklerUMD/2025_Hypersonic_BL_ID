function [starts,stops] = checkOverlap(starts,stops,BL_height,cone,image_ft)
%% [starts,stops] = checkOverlap(starts,stops,BL_height,cone,image_ft)
%
%   Inputs:
%       starts - the start locations for the wave packets in the current
%           image
%       stops - the stop locations for the same wave packets
%       BL_height - the row where the top of the boundary layer is
%           suspected to be
%       cone - the highest (smallest index) row where the cone is intruding
%           into the image
%       image_ft - the Image that has been prepared for the fourier
%           transform, meaning the averaged image has already been
%           subtracted out.
%
%   Outputs:
%       starts - the modified starts that has grouped any that are
%           considered overlaping
%       stops - the modified starts with the same groupings
%
%   This function uses windowing methods to determine if overlaping
%   indications of wave packets are actually marking the same packet. If
%   the maximum of the fourier transform in the range between the start of
%   one packet and the end of the one in question doesn't drop below 50% of
%   the maximum for the entire interval then the packets are considered to
%   be overlaping and are combined.
%


% a figure just in case
% h = figure('Name','Window Testing');
% pause on

% calculate the BL thickness
BL_thickness = cone-BL_height;

% set the number of points
pts = 2^16;

% the frequencies of interest
f = (0:pts/2)/pts;
% Frequency bounds for 2nd mode frequency
find_min = find(f>1/(4.5*BL_thickness),1,'first');
find_max = find(f<(1/(BL_thickness*1.5)),1,'last');

% get the starts and stops sorted
if ~isrow(starts)
    starts = starts';
end
if ~isrow(stops)
    stops = stops';
end
% make sure the starts and stops are in order
Temp = [starts',stops'];
Temp = sortrows(Temp,1);
starts = Temp(:,1);
stops = Temp(:,2);

for i = 1:max(size(starts))-1
    windowSize = ceil(stops(i))-floor(starts(i));
    for j = i+1:max(size(starts))
        % check to see if the other location is nearby (within 50% of the
        % current windowSize)
        if starts(j) < stops(i)+windowSize/2
            % if the other one is smaller use it's size
            if stops(j)-starts(j) < windowSize
                windowSize = ceil(stops(j)-starts(j));
            end
            % create the window
            x = 0:windowSize;
            window = (1-0.16)/2-(1/2)*cos(2*pi*x/(windowSize-1))+0.16/2*...
                cos(4*pi*x/(windowSize-1));
            window = ones(size(image_ft,1),1)*window;
            % figure out how many steps to take between the start of the
            % ith section and the jth section and set up an array to store
            % the maxes.
            steps = floor((floor(stops(j)-windowSize)-floor(starts(i)))/(3*BL_thickness));
            maxes = zeros(1,steps);
            for k = 0:steps
                tempInd = floor(starts(i)) + 3*k*BL_thickness;
                if tempInd+windowSize > size(image_ft,2)
                    tempInd = size(image_ft,2)-windowSize;
                end
                temp = fft(image_ft(BL_height:cone,tempInd:tempInd+windowSize).*...
                    window(BL_height:cone,:),pts,2);
                wFT{k+1} = temp.*conj(temp)/pts;
                % find the peak in the frequency range of interest for 2nd mode
                % packets
                maxes(k+1) = max(max(wFT{k+1}(:,find_min:find_max)));
            end
            if min(maxes) > 0.5*max(maxes)
                Temp(i,2) = Temp(j,2);
                Temp(j,:) = 0;
            end
        end
    end
end
% set the new starts and stops
starts = Temp(Temp(:,1)>0,1);
stops = Temp(Temp(:,2)>0,2);

% pause off


end