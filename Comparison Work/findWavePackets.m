function [packet_bounds] = findWavePackets(Data_filtered2nd,Data_filtered1h,BL_thickness,Data_raw,turbulentImage,varargin)
% function [packet_bounds] = findWavePackets(Data_filtered2nd,Data_filtered1h,BL_thickness,Data_raw,turbulentImage)
%
%   Input:
%       Data_filtered2nd - data that has been bandpass filtered around the
%           second mode
%       Data_filtered1h - data that has been bandpass filtered around the
%           first harmonic
%       BL_thickness - the thickness of the boundary layer in pixels
%       Data_raw - the unfiltered data, used to determine the minimum
%           allowable height
%       turbulentImage - 1 if the image is indicated as having a
%           significant turbulent spot 0 otherwise
%
%   Output:
%       packet_bounds - the boundaries on the packets, in terms of column
%       indexes. This is a three dimensional array that is indexed by
%       (row,packet_number,L/R bound).
%
%   This function determines the extent of any wave packets in the data
%   passed in by looking at the individual rows in Data_filtered2nd, which
%   is assumed to be filtered for the 2nd mode. Packets are determined by
%   where there are peaks that are seperated by the correct distance
%   (sepMin and sepMax define the range) and must initially grow, then stay
%   roughly the same, and then shrink. This function also checks to make
%   sure that there is not an overwhelming amount of 1st harmonic in the
%   area where the wave packet is expected to be. If there is then the spot
%   is presumed to be turbulent. The peaks must also exceed some minimum
%   height that is determined by using information about the standard
%   diviation of the origional data.
%

% bounds on separation (the peak separation must be between these values)
% this is based on the assumption that the true boundary layer thickness
% is 1.5 times thicker than the measured boundary layer thickness based on
% density
sepMax = 3*BL_thickness;
sepMin = 1.8*BL_thickness;
% limit on the peak to peak variation
peak_difference = 0.05; % magnitude difference as a percentage of the maximum value
peak_variation = 0.9; % relative difference p2/p1

% this checks if you want to use the various thresholds that are in place
% to make sure that a laminar boundary layer or turbulent spot is not
% incorrectly identified as a second-mode wave packet
yesThresholds = max(max(Data_filtered1h))~=0;

% initialize packet_bounds
packet_bounds = zeros(BL_thickness+1,1,2);
for r = 1:size(Data_filtered2nd,1)
    
    % Since this method uses peak finding the data needs to be smoothed so
    % that there aren't to many double peaks
    correlation = smoothdata(Data_filtered2nd(r,:));
    [peak,loc] = findpeaks(correlation);
    [peakh,loch] = findpeaks(smoothdata(Data_filtered1h(r,:)));
    if length(peak) >= 5
        % check the variation in the peak separation
        separation_stdv = zeros(1,length(peak)-4);
        for i = 1:length(peak)-4
            separation = loc(i+1:i+3)-loc(i:i+2);
            separation_stdv(i) = std(separation);
        end

        % limit on the variance of the peak seperation
        stdev_limit = (sepMax-sepMin)/4;
        
        current_packet = [0,0];
        ind = 1;
        for i = 1:length(peak)-4
            if loc(i) > current_packet(1) && loc(i) < current_packet(2)
                % Do nothing if you are a peak that is already in a wave pakcet
            else
                
                % set up variables to be used and the magnitude threshold
                separation = loc(i+1:i+3)-loc(i:i+2);
                magMin = 0.25*std(Data_raw(r,loc(i):loc(i+3)))*yesThresholds;
                
                % check the standard diviation in the separation and the
                % average peak seperation and make sure that some of the
                % peaks are big
                if (separation_stdv(i) < stdev_limit) && ...
                        (mean(separation) < sepMax) && ...
                        (mean(separation) > sepMin) && ...
                        (sum(peak(i:i+3)>magMin) > 2)
                    % set up variables to control logic in while loop
                    isPacket = 1; % still going in the loop
                    upDown = 1; % beginning middle or end of the packet 
                                % (this might get eliminated)
                    j = 0; % counter
                    if loc(i+j)-3*BL_thickness >1
                        current_packet(1) = loc(i+j)-3*BL_thickness; % the boundaries of the current packet
                    else
                        current_packet(1) = 1; % the boundaries of the current packet
                    end
                    
                    up = []; % place to store the peaks in the increasing part of the packet
                    while isPacket>0
                        if i+j+1 > length(loc)
                            % if you've reached the end of the image
                            break
                        end
                        % check separation
                        if (loc(i+j+1)-loc(i+j)) < sepMax && ...
                                (loc(i+j+1)-loc(i+j)) > sepMin
                            % check peak differences
                            peakDiff = peak(i+j+1)-peak(i+j);
%                             seperation = loc(i+j+1)-loc(i+j);
                            maxPeak = max(peak(i+j+1),peak(i+j));
                            
                            % logic based on a iIIi structure of a wave
                            % packet
                            if upDown == 1
                                % This is for the beginning of the packet
                                % where the peaks should be increasing
                                % noticably
                                up(j+1) = peakDiff;
                                % if it dips for one peak and then keeps
                                % going up
                                if up(j+1) < -peak_difference*maxPeak &&...
                                        length(peak) > i+j+2 
                                    if ((loc(i+j+2)-loc(i+j)) < sepMax && ...
                                        (loc(i+j+2)-loc(i+j)) > sepMin) || ...
                                        peak(i+j+2)-peak(i+j)>0
                                        up(j+1) = peak(i+j+2)-peak(i+j);
                                    end
                                end
                                upDown = upDown - ~(mean(up)>...
                                    peak_difference*maxPeak);
                            elseif upDown == 0
                                % The middle where the peaks are relatively
                                % constant
                                upDown = upDown - ~(peak(i+j+1)/peak(i+j) > peak_variation) ...
                                    - (peak(i+j+1) < magMin);
                            else
                                % The end where the peaks should be decreasing
                                if ~(peakDiff < -0.4*maxPeak) || ...
                                        peak(i+j+1)< magMin
                                    isPacket = -1;
                                end
                            end
                            j=j+1;
                        else
                            isPacket = -1;
                            if j == 0
                                current_packet(1) = 0;
                            end
                        end
                    end
                    % Boundaries of the current packet
                    if j ~= 0
                        if length(Data_filtered1h) > loc(i+j)+3*BL_thickness
                            current_packet(2) = loc(i+j)+3*BL_thickness;
                        else
                            current_packet(2) = length(Data_filtered2nd);
                        end
                    end
                    
                    % turbulence threshold: if the 1st harmonic has a
                    % similar or higher amplitude then it's probably
                    % turbulent or weak.
                    localPeaks = peakh(((loch>loc(i)-3*BL_thickness).*...
                        (loch<loc(i+j)+3*BL_thickness))>0);
                    % check to make sure that the variation in the filtered
                    % signal corrosponds to variation in the unfiltered
                    % signal
                    stdCheck = (std(Data_filtered2nd(r,loc(i):loc(i+j))) > ...
                        std(Data_raw(r,loc(i):loc(i+j)))/2*yesThresholds);
                    if isempty(localPeaks)
                        localPeaks = 0;
                    end
                    if isempty(stdCheck)
                        stdCheck = 1;
                    end
                    
                    % save the wave packet boundaries
                    if j > 2 && sum(max(localPeaks)*(1.25+1*...
                            turbulentImage)>peak(i:i+j))<=(j+1)/3 &&...
                            (mean(peak(i:i+j)>(1.25+turbulentImage)...
                            *mean(localPeaks))) && stdCheck
                        % if the packet doesn't have significant detectable
                        % turbulence
                        
                        packet_bounds(r,ind,1) = current_packet(1);
                        packet_bounds(r,ind,2) = current_packet(2);
                        ind = ind+1;
                        % move along and check the next thing
                        
                        % plotting the different steps of the process
                        if r == 5 && ~isempty(varargin)
                            figure % plot the filtered data with the identified peaks
                            plot(1:length(Data_raw),Data_filtered2nd(r,:),'b-',...
                                loc(i:i+j),Data_filtered2nd(r,loc(i:i+j)),'rx')%,...
%                                 ones(1,floor(max(Data_filtered2nd(r,:))-min(Data_filtered2nd(r,:)))+1)*current_packet(1),min(Data_filtered2nd(r,:)):max(Data_filtered2nd(r,:)),'--',...
%                                 ones(1,floor(max(Data_filtered2nd(r,:))-min(Data_filtered2nd(r,:)))+1)*current_packet(2),min(Data_filtered2nd(r,:)):max(Data_filtered2nd(r,:)),'--',...
%                                 1:length(Data_raw),ones(1,length(Data_raw))*magMin,'--')
                            set(gca,'FontName','Times New Roman','FontSize',18)
                            xlabel('Location Along Image (pixels)','FontSize',18,'FontName','Times New Roman')
                            xlim([1 length(Data_raw)])
                            ylabel('Intensity','FontSize',18,'FontName','Times New Roman')
                            figure % add in the magnitude limit
                            plot(1:length(Data_raw),Data_raw(r,:),'-','Color',[0.75 0.75 0])
                            hold on
                            plot(1:length(Data_raw),Data_filtered2nd(r,:),'b-',...
                                loc(i:i+j),Data_filtered2nd(r,loc(i:i+j)),'rx',...)%
                                1:length(Data_raw),ones(1,length(Data_raw))*magMin,'k--')%,...
                            hold off
%                                 ones(1,floor(max(Data_filtered2nd(r,:))-min(Data_filtered2nd(r,:)))+1)*current_packet(1),min(Data_filtered2nd(r,:)):max(Data_filtered2nd(r,:)),'g--',...
%                                 ones(1,floor(max(Data_filtered2nd(r,:))-min(Data_filtered2nd(r,:)))+1)*current_packet(2),min(Data_filtered2nd(r,:)):max(Data_filtered2nd(r,:)),'m--',...
                            set(gca,'FontName','Times New Roman','FontSize',18)
                            xlabel('Location Along Image (pixels)','FontSize',18,'FontName','Times New Roman')
                            legend('Unifiltered','Filtered around second mode')
                            xlim([1 length(Data_raw)])
                            ylabel('Intensity','FontSize',18,'FontName','Times New Roman')
                            figure % take out the magnitude limit and put in the 1h signal for comparison
                            plot(1:length(Data_raw),Data_filtered1h(r,:),'-','Color',[0.75 0 0.75])
                            hold on
                            plot(1:length(Data_raw),Data_filtered2nd(r,:),'b-',...
                                loc(i:i+j),Data_filtered2nd(r,loc(i:i+j)),'rx')%)
%                                 ones(1,floor(max(Data_filtered2nd(r,:))-min(Data_filtered2nd(r,:)))+1)*current_packet(1),min(Data_filtered2nd(r,:)):max(Data_filtered2nd(r,:)),'g--',...
%                                 ones(1,floor(max(Data_filtered2nd(r,:))-min(Data_filtered2nd(r,:)))+1)*current_packet(2),min(Data_filtered2nd(r,:)):max(Data_filtered2nd(r,:)),'m--',...
                            hold off
                            set(gca,'FontName','Times New Roman','FontSize',18)
                            xlabel('Location Along Image (pixels)','FontSize',18,'FontName','Times New Roman')
                            xlim([1 length(Data_raw)])
                            legend('Filtered around first harmonic','Filtered around second mode')
                            ylabel('Intensity','FontSize',18,'FontName','Times New Roman')
                            figure % take out 1h signal and put in the packet bounds
                            plot(1:length(Data_raw),Data_filtered2nd(r,:),'b-',...
                                loc(i:i+j),Data_filtered2nd(r,loc(i:i+j)),'ro')%,...
                            hold on
                            plot(ones(1,floor(max(Data_filtered2nd(r,:))-min(Data_filtered2nd(r,:)))+1)*current_packet(1),min(Data_filtered2nd(r,:)):max(Data_filtered2nd(r,:)),'--','Color',[0 0.5 0])
                            plot(ones(1,floor(max(Data_filtered2nd(r,:))-min(Data_filtered2nd(r,:)))+1)*current_packet(2),min(Data_filtered2nd(r,:)):max(Data_filtered2nd(r,:)),'--','Color',[0.75 0 0.75])
                            hold off
                            set(gca,'FontName','Times New Roman','FontSize',18)
                            xlabel('Location Along Image (pixels)','FontSize',18,'FontName','Times New Roman')
                            xlim([1 length(Data_raw)])
                            ylabel('Intensity','FontSize',18,'FontName','Times New Roman')
                        end
                    else
                        current_packet = [0,0];
                    end
                end
            end
        end
    end
end
end