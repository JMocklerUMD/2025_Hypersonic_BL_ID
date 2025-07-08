function [centers] = findCenters(peaks,locs,BL_thickness,totalSpread)
%%
%
%

minSupport = totalSpread/8;
peaks = peaks/max(peaks); % normalize so that thresholding is easier
minSeparation = 6*BL_thickness; % minimum seperation between voter and average

notUsed = ones(size(locs));
centers = [];
tind = 1;
for i = 1:length(locs)
    if notUsed(i) == 0
        % if you're already used don't do anything
    else
        % if not pretend like you're an average and find out where the
        % center should be
        cmean = locs(i);
        notDone = 1;
        while notDone
            cSupporters = find(locs>cmean-minSeparation & locs<cmean+minSeparation & notUsed);
            temp = weightedAverage(locs(cSupporters),peaks(cSupporters));
            if temp==cmean
                notDone = 0;
                if sum(peaks(cSupporters))>minSupport
                    centers(tind) = cmean;
                    notUsed(cSupporters) = 0;
                    tind = tind+1;
                end
            else
                cmean = temp;
            end
        end
    end
end

end