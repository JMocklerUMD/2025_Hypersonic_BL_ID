function [IM,nn] = referenceImage(no_ref,furthest_image,img_no,base_dir,shot,imgPrev,nnPrev,IM)
% [IM,nn] = referenceImage(no_ref,furthest_image,img_no,base_dir,shot,imgPrev,nnPrev,IM)
%
%   Inputs:
%       no_ref - the number of images to include in the reference image
%           both before and after the current image. Thus the total number
%           of images averaged into the sequence is no_ref*2+1.
%       furthest_image - the last image in the sequence
%       img_no - the current image number
%       base_dir - the full path to the directory where the images folder
%           is
%       shot - the shot number for this image sequence
%       imgPrev - the number of the previous image, if there was no
%           previous image then this value should be 0.
%       nnPrev - the number of images that went into making the previous
%           reference image. This should also be 0 if there was no previous
%           image.
%       IM - the previous reference image, if there was one.
%
%   Outputs:
%       IM - the current reference image
%       nn - the current number of images that went into creating the
%           reference image
%
% This function builds a refference image by taking images from,
% image_no-no_ref to image_no+no_ref, and averaging them. If this range
% exceeds the avaliable images it is truncated to the images avaliable and
% the number of images is changed accordingly. 
%

% find the correct function to read the images
imgReadFxn = findImageReadFxn(shot);

% parameters for reference image
if img_no+no_ref<furthest_image && img_no-no_ref>0
    ref_range = img_no-no_ref:img_no+no_ref;
% elseif img_no+no_ref<=furthest_image
%     ref_range = 1:img_no+no_ref;
else
    ref_range = img_no-no_ref:furthest_image;
end
nn = length(ref_range);

% depending on the difference between the last image and this one, change
% the refference image to be the one for the current image
if abs(img_no-imgPrev) > no_ref || imgPrev == 0
    % The current image is beyond the range of the previous refference
    % image so make a new one
    for ii = 1:length(ref_range)
        
        %logic in case the expected image doesn't exist
        try
            I = imgReadFxn(base_dir,shot,ref_range(ii));
        catch
            nn = nn - 1;
            continue;
        end
        
        
        if ii == 1
          IM = double(I);
        else
          IM = IM + double(I);
        end
    end
    IM = IM/nn;
    
elseif img_no-imgPrev > 0
    % the current image is further along in the sequence than the previous
    % image
    IM = IM*nnPrev/nn;
    % subtract out images that are no longer needed
    for ii = imgPrev-no_ref:img_no-no_ref-1
        if ii >=1 && ii<=furthest_image
            I = imgReadFxn(base_dir,shot,ii);
            IM = IM - double(I)/nn;
        end
    end
    % add in new images
    for ii = imgPrev+no_ref+1:img_no+no_ref
        if ii >=1 && ii<=furthest_image
            I = imgReadFxn(base_dir,shot,ii);
            IM = IM + double(I)/nn;
        end
    end
    
else
    % if the current image number is less than the previous one
    IM = IM*nnPrev/nn;
    % add in new images
    for ii = img_no-no_ref:imgPrev-no_ref-1
        if ii >=1 && ii<=furthest_image
            I = imgReadFxn(base_dir,shot,ii);
            IM = IM + double(I)/nn;
        end
    end
    % subtract out old ones
    for ii = img_no+no_ref+1:imgPrev+no_ref
        if ii >=1 && ii<=furthest_image
            I = imgReadFxn(base_dir,shot,ii);
            IM = IM - double(I)/nn;
        end
    end
end