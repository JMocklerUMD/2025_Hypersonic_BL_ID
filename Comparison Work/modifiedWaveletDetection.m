function [secondMode] = modifiedWaveletDetection(base_dir,shot,loopRange,furthest_image,no_ref,thresholdShift,display)
%% [secondMode] = modifiedWaveletDetection(base_dir,shot,loopRange,furthest_image,no_ref,thresholdShift,display)
%
%   Inputs:
%       base_dir - the current directory, or the directory in which the
%           other relevant folders are
%       shot - the number identifying the current image sequence
%       loopRange - the list of images to process
%       furthest_image - the last image in the sequence
%       no_ref - half the number of images to be used in creating the
%           averaged image
%       thresholdShift - a modification to the threshold used for
%           turbulence detection
%       display - a logical value used to determine whether to display
%           results or not
%
%   Output:
%       secondMode - a vector of the same length as loopRange that
%           indicates which images have second mode wave packets.
%
% This function identifies images that have second-mode wave packets using
% the wavelet transform and the modified wavelet outlined in AIAA
% 2015-1787. There are several parameters inside of this function must be
% modified to produce accurate results.
%

global loopCount rowShift moveEvent changeA
addpath([base_dir '\wave_packet_detection']);
% get the correct function to read the images with
imgReadFxn = findImageReadFxn(shot);
loopCount = 1;

% stuff to control 2nd mode identification
frequencyFactor = [1 1.5 2 2.5 3];
minSpread = 2; % minimum spread of an ideal packet
maxSpread = 30; % maximum spread of an ideal packet
secondMode = zeros(size(loopRange)); % store information on whether there are 2nd mode wave packets or not
verificationFactor = 0.4/10; % THIS MUST BE MODIFIED
verificationWidthPercent = 0.2;
verificationHeightNumber = 6;
minAThreshold = 0.1; % THIS MUST BE MODIFIED

% stuff for trubulence analysis
TspotOverlap = 0.3; % for checking with the C results
widthFactor = 3;

% create the averaged image and save the number of images used to create it
[IM,nnPrev] = referenceImage(no_ref,furthest_image,loopRange(1),base_dir,shot,0,0,0);
IMprev = IM;
imgPrev = loopRange(loopCount);

% if displaying make a figure
if display>0
    h = figure('KeyPressFcn',@figureFunction);
end

while loopCount <= length(loopRange)
    rowShift = 0; % shift variable for display
    moveEvent = 0;
    img_no = loopRange(loopCount);
    % create the averaged image and save the number of images used to create it
    [IM,nnPrev] = referenceImage(no_ref,furthest_image,img_no,base_dir,shot,imgPrev,nnPrev,IMprev);
    IMprev = IM;
    imgPrev = loopRange(loopCount);
    imgLength = length(IM);
    % read in image file, rotate
    [Img,BLmethod,rotate] = imgReadFxn(base_dir,shot,img_no);
    Img = double(Img);
    image_ft =(Img-IM); 
    % now if you need to rotate the images do so
    if rotate
        % rotate based on averaged image
        [IM,yp,rowInds,colInds] = coneAdjuster(IM);
        temp = fit((1:length(yp))',yp','poly1');
        ang = atan(temp.p1);
        Img = imrotate(Img,ang*180/pi,'bilinear');
        image_ft = imrotate(image_ft,ang*180/pi,'bilinear');
        Img = Img(rowInds,colInds);
        image_ft = image_ft(rowInds,colInds);
        tempRows = ceil(yp(end)):size(IM,1);
        [cone,BL_height,BL_thickness] = findBL(IM(tempRows,:),BLmethod);
        cone = cone+ceil(yp(end))-1;
        BL_height = BL_height+ceil(yp(end))-1;
        imgLength = length(image_ft);
        if length(IM) ~= imgLength
            disp('Something is wrong')
        end
    else
        [cone,BL_height,BL_thickness] = findBL(IM,BLmethod); % find information about the boundary layer
    end
    
    % run a band pass filter over the boundary layer 
    if BL_thickness >= 3.2/1.5
        bp_filter = fdesign.bandpass(1.25/(4.5*BL_thickness),(1.5)/(4.5*BL_thickness),...
            2/(1*BL_thickness),(3.2)/(1.5*BL_thickness),10,2,10);
    else
        bp_filter = fdesign.bandpass(1.25/(4.5*BL_thickness),(1.5)/(4.5*BL_thickness),...
            0.9,1,10,2,10);
    end
    bp_filter = design(bp_filter,'cheby1');
    filtered_img = zeros(BL_thickness+1,imgLength);
    for tind = BL_height:cone
        filtered_img(tind-BL_height+1,:) = filter(bp_filter,image_ft(tind,:));
    end
    
    % initialize variables
    wave_packets = zeros(BL_thickness+1,1,2);
    resultH1 = zeros(maxSpread-minSpread+1,imgLength);
    resultH2 = resultH1;
    resultC1 = resultH2;
    resultC2 = resultH2;
    resultC3 = resultH2;
    normalizedImage_ft = zeros(cone,imgLength);
    
    % do an initial loop to hopefully detect turbulent spots and save
    % results for confirming wave packets later
    for shiftUp = 0:cone-1
        % Modified wavelet analysis on normalized rows starting by
        % normalizing the rows
        normalizedRow = (image_ft(cone-shiftUp,:))/...
            (sqrt(sum(image_ft(cone-shiftUp,:).^2)));
        normalizedImage_ft(cone-shiftUp,:) = normalizedRow;        

        % preallocate
        resultC = zeros(maxSpread-minSpread+1,imgLength,length(frequencyFactor));
        
        % go through the different "Frequencies"
        for j = 1:length(frequencyFactor)
            if j < 3 || shiftUp <= BL_thickness
            % different spreads
            for spread = minSpread*BL_thickness:BL_thickness:maxSpread*BL_thickness
                % create the ideal packet
                x = -spread-2*BL_thickness:spread+2*BL_thickness;
                packetModel = idealPacket(BL_thickness,spread,x,0,frequencyFactor(j));
                
                % using built in convolution
                resultC(spread/BL_thickness-1,:,j) = abs(conv(normalizedRow,...
                    packetModel/sqrt(sum(packetModel.*conj(packetModel))),'same'));
            end
            end
        end 
        
        resultH1 = resultH1+resultC(:,:,1);
        resultH2 = resultH2+resultC(:,:,2);
        if shiftUp<=BL_thickness
            resultC1 = resultC1+resultC(:,:,3);
            resultC2 = resultC2+resultC(:,:,4);
            resultC3 = resultC3+resultC(:,:,5);
        end
    end
    
    % check for turbulent spots using 1st harmonic and 2nd mode data
    turbulentSpots = waveletTurbulence(resultH1,BL_thickness,maxSpread,minSpread,thresholdShift);
    temp = waveletTurbulence(resultH2,BL_thickness,maxSpread,minSpread,thresholdShift);
    if turbulentSpots(1,1) == 0
        turbulentSpots = temp;
    elseif temp(1,1) ~= 0
        turbulentSpots(size(turbulentSpots,1)+1:size(turbulentSpots,1)+...
            size(temp,1),:) = temp;
    end
    % for the C checks they must agree with one other in order to be
    % counted
    tempTurb = waveletTurbulence(resultC1,BL_thickness,maxSpread,minSpread,thresholdShift);
    
    temp = waveletTurbulence(resultC2,BL_thickness,maxSpread,minSpread,thresholdShift);
    if tempTurb(1,1) == 0
        tempTurb = temp;
    elseif temp(1,1) ~= 0
        tempTurb(size(tempTurb,1)+1:size(tempTurb,1)+...
            size(temp,1),:) = temp;
    end
    temp = waveletTurbulence(resultC3,BL_thickness,maxSpread,minSpread,thresholdShift);
    if tempTurb(1,1) == 0
        tempTurb = temp;
    elseif temp(1,1) ~= 0
        tempTurb(size(tempTurb,1)+1:size(tempTurb,1)+...
            size(temp,1),:) = temp;
    end
    
    if turbulentSpots(1,1) ~= 0
    for k = 1:size(tempTurb,1)
        left = tempTurb(k,1);
        right = tempTurb(k,2);
        added = 0;
        checkSpots = tempTurb([1:k-1 k+1:end],:);
        % check for overlap on the left side
        checkSpots = [turbulentSpots(turbulentSpots(:,2)>left & ...
            turbulentSpots(:,2) < right,:); checkSpots(checkSpots(:,2)>left & ...
            checkSpots(:,2)<right,:)];
        if ~isempty(checkSpots)
            if sum(checkSpots(:,2)-left > TspotOverlap*(checkSpots(:,2)-checkSpots(:,1)))>0
                turbulentSpots(size(turbulentSpots,1)+1,:) = [left,right];
                added = 1;
            end
        end
        checkSpots = tempTurb([1:k-1 k+1:end],:);
        % and on the right side
        checkSpots = [turbulentSpots(turbulentSpots(:,1)<right & ...
            turbulentSpots(:,1) > left,:); checkSpots(checkSpots(:,1)<right & ...
            checkSpots(:,1)>left,:)];
        if ~isempty(checkSpots) && ~added
            if sum(right-checkSpots(:,1) > TspotOverlap*(checkSpots(:,2)-checkSpots(:,1)))>0
                turbulentSpots(size(turbulentSpots,1)+1,:) = [left,right];
                added = 1;
            end
        end
        checkSpots = tempTurb([1:k-1 k+1:end],:);
        % and if you're encased by another spot
        checkSpots = [turbulentSpots(turbulentSpots(:,1)<left & ...
            turbulentSpots(:,2)>right,:); checkSpots(checkSpots(:,1)<left & ...
            checkSpots(:,2)>right,:)];
        if ~isempty(checkSpots) && ~added
            turbulentSpots(size(turbulentSpots,1)+1,:) = [left,right];
            added = 1;
        end
        checkSpots = tempTurb([1:k-1 k+1:end],:);
        % or if you encase another spot
        checkSpots = [turbulentSpots(turbulentSpots(:,1)>left & ...
            turbulentSpots(:,2)<right,:); checkSpots(checkSpots(:,1)>left & ...
            checkSpots(:,2)<right,:)];
        if ~isempty(checkSpots) && ~added
            turbulentSpots(size(turbulentSpots,1)+1,:) = [left,right];
        end
    end
    end
    
    % fourier transform for weighting and turbulence detection
    FT = fft(image_ft(BL_height:cone,:),2^12,2);
    FT = FT.*conj(FT);
    f = (1:2^12)/2^12;
    % Frequency bounds for 2nd mode frequency
    find_min = find(f>1/(4.5*BL_thickness),1,'first');
    find_max = find(f<(3/(BL_thickness*1.5)),1,'last');
    weighting = max(FT(:,find_min:find_max),2);
    weighting = weighting/max(weighting);
    
    calmColumns = ones(1,imgLength);
    for k = 1:size(turbulentSpots,1)
        calmColumns = calmColumns - (1:imgLength>turbulentSpots(k,1) & 1:imgLength<turbulentSpots(k,2));
    end
    
    resultA = zeros(maxSpread-minSpread+1,imgLength,BL_thickness+1,3);
    tInd = 1;
    for i = 1:3
    for shiftUp = 0:BL_thickness
        
        normalizedRowF = (filtered_img(end-shiftUp,:))/...
            sqrt(sum(filtered_img(end-shiftUp,:).^2));
        
        for spread = minSpread*BL_thickness:BL_thickness:maxSpread*BL_thickness
            % create the ideal packet
            x = -spread-2*BL_thickness:spread+2*BL_thickness;
            packetModel = idealPacket(BL_thickness,spread,x,0,frequencyFactor(end-i+1));
            % If I remember correctly this is only done once
            % because it looks the same for every other combination
            for center = 1:imgLength
                xt = x+center;
                resultA(spread/BL_thickness-1,center,BL_thickness+1-shiftUp,i) = abs(sum(...
                    normalizedRowF(xt(xt>0 & xt<= imgLength)).*packetModel(...
                    (xt>0 & xt<= imgLength))/sqrt(sum(packetModel(xt>0 & ...
                    xt<=imgLength).*conj(packetModel(xt>0 & xt<=imgLength))))));
            end
        end
        
        % new (7/3/14) code for locating wave packets based on resultA
        bwA = resultA(1:ceil((maxSpread-minSpread)/2),:,BL_thickness+1-shiftUp,i)>std(normalizedRowF(calmColumns==1)) & ...
            resultA(1:ceil((maxSpread-minSpread)/2),:,BL_thickness+1-shiftUp,i)>minAThreshold;
        bwAlabeled = bwlabel(bwA);
        notCovered = ones(1,max(max(bwAlabeled)));
        
        for j = 1:length(notCovered)
            if notCovered(j)
                peak = max(max(resultA(1:ceil((maxSpread-minSpread)/2),:,BL_thickness+1-shiftUp,i).*(bwAlabeled==j))); % get the maximum value
                [row,col] = find((resultA(1:ceil((maxSpread-minSpread)/2),:,BL_thickness+1-shiftUp,i).*(bwAlabeled==j))==peak,1); % and it's location
                [goodWedge,lefts,rights,tempBW] = wedgeAnalyzer(resultA(:,:,BL_thickness+1-shiftUp,i),peak,row,col,BL_thickness,1);
                
                
                if goodWedge
                    for k = j+1:length(notCovered)
                       if sum(sum(tempBW(1:ceil((maxSpread-minSpread)/2),:).*(bwAlabeled==k)))>0
                           notCovered(k) = 0;
                       end
                    end
                    
                    turbulentOverlap = 0;
                    left = mean(lefts);
                    right = mean(rights);
                    for k = 1:size(turbulentSpots,1)
                        % for each turbulent spot make sure that the packet
                        % has less than 20% overlap
                        turbulentOverlap = turbulentOverlap || ...
                            ((right-turbulentSpots(k,1))/(right-left)>0.2...
                             && left<turbulentSpots(k,1))|| ...
                            ((turbulentSpots(k,2)-left)/(right-left) > 0.2 && ...
                            right>turbulentSpots(k,2)) || ...
                            (right<turbulentSpots(k,2) && ...
                            left>turbulentSpots(k,1));
                    end
                    
                    % currently only checking for overlap
                    % now also checking for overlap with turbulence
                    if ~(sum(mean(lefts)<wave_packets(end-shiftUp,:,1))>0 &&...
                            sum(mean(rights)>wave_packets(end-shiftUp,:,2))>0)...
                            && ~turbulentOverlap && mean(rights)-mean(lefts) > widthFactor*BL_thickness
                        wave_packets(end-shiftUp,tInd,1) = mean(lefts);
                        wave_packets(end-shiftUp,tInd,2) = mean(rights);
                        tInd = tInd+1;
                    end

                end
                notCovered(j) = 0;
            end
        end
    end
    end
    % here's where post processing of wave_packets should happen
    weighting = weighting/max(weighting);
    % find the wave packets and their boundaries (same way as
    % frequencyIdentification
    starts = [];
    stops = [];
    if sum(wave_packets(:,1,1)) > 0
        [starts,stops] = findBounds(wave_packets,weighting,BL_thickness);
        if ~isempty(starts) || ~isempty(stops)
            pairs = pairBounds(starts,stops,wave_packets,BL_thickness,weighting);
            tinds = [];
            for j = 1:size(pairs,1)
                % check against the true convolution to determine if
                % there's actually anything there
                lowThreshold = (BL_thickness+1)*verificationFactor;
                halfway = floor((maxSpread-minSpread+1)/2);
                
                left = floor(starts(pairs(j,1)));
                right = ceil(stops(pairs(j,2)));
                if right > left
                    lRange = left:right;
                    lRange = lRange(lRange>0 & lRange <= imgLength);
                    
                    % check against each of the C results
                    peak = max(max(resultC1(:,lRange)));
                    tempBW = resultC1>0.8*peak;
                    tempBW(:,[1:lRange(1), lRange(end):imgLength]) = 0;
                    labeledBW = bwlabel(tempBW);
                    for l = 1:max(max(labeledBW))
                        temp = labeledBW == l;
                        if sum(sum(temp(halfway+1:end,:)))>2/3*sum(sum(temp))
                            tempBW(temp) = 0;
                        end
                    end
                    goodWedge = max(sum(tempBW,2))>(lRange(end)-lRange(1))*verificationWidthPercent && ...
                        max(sum(tempBW,1))> verificationHeightNumber;
                    verifiedWedge = goodWedge && peak > lowThreshold;
                    if ~ verifiedWedge
                        % check against each of the C results
                        peak = max(max(resultC2(:,lRange)));
                        tempBW = resultC2>0.8*peak;
                        tempBW(:,[1:lRange(1), lRange(end):imgLength]) = 0;
                        labeledBW = bwlabel(tempBW);
                        for l = 1:max(max(labeledBW))
                            temp = labeledBW == l;
                            if sum(sum(temp(halfway+1:end,:)))>2/3*sum(sum(temp))
                                tempBW(temp) = 0;
                            end
                        end
                        goodWedge = max(sum(tempBW,2))>(lRange(end)-lRange(1))*verificationWidthPercent && ...
                            max(sum(tempBW,1))> verificationHeightNumber;
                        verifiedWedge = goodWedge && peak > lowThreshold;
                        if ~ verifiedWedge
                            % check against each of the C results
                            peak = max(max(resultC3(:,lRange)));
                            tempBW = resultC3>0.8*peak;
                            tempBW(:,[1:lRange(1), lRange(end):imgLength]) = 0;
                            labeledBW = bwlabel(tempBW);
                            for l = 1:max(max(labeledBW))
                                temp = labeledBW == l;
                                if sum(sum(temp(halfway+1:end,:)))>2/3*sum(sum(temp))
                                    tempBW(temp) = 0;
                                end
                            end
                            goodWedge = max(sum(tempBW,2))>(lRange(end)-lRange(1))*verificationWidthPercent && ...
                                max(sum(tempBW,1))> verificationHeightNumber;
                            verifiedWedge = goodWedge && peak > lowThreshold;
                        end
                    end
                    if verifiedWedge
                        tinds = [tinds,j];
                    end
                end
            end
            if isempty(tinds)
                starts = [];
                stops = [];
            else
                starts = starts(pairs(tinds,1));
                stops = stops(pairs(tinds,2));
                secondMode(loopCount) = 1;
            end
        end
    end
    
    if display == 1
      pause on
        Aind = 3;
        while ~moveEvent
            displayWavelet(h,Img,image_ft,img_no,cone,BL_height,wave_packets,...
                starts,stops,resultA(:,:,:,Aind),0.25,minSpread:maxSpread,resultC1,...
                (BL_thickness+1)/10,resultC2,(BL_thickness+1)/10,resultC3,(BL_thickness+1)/10);
            pause()
            Aind = Aind+changeA;
            if Aind>size(resultA,4)
                Aind = size(resultA,4);
            elseif Aind < 1
                Aind = 1;
            end
            loopCount = loopCount+1;
        end
    elseif display == 2
        Aind = 3;
        pause on
        while ~moveEvent
            displayWavelet(h,Img,image_ft,img_no,cone,BL_height,wave_packets,...
                starts,stops,resultA(:,:,:,Aind),0.25,minSpread:maxSpread,...
                resultH1,(BL_thickness+1)/10,resultH2,(BL_thickness+1)/10,turbulentSpots);
            pause()
            Aind = Aind+changeA;
            if Aind>size(resultA,4)
                Aind = size(resultA,4);
            elseif Aind < 1
                Aind = 1;
            end
            loopCount = loopCount+1;
        end
    else
        loopCount = loopCount+1;
    end
    
    if loopCount<1
        loopCount=1;
    end
    
end


end