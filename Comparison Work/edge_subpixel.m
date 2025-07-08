function [xe_sub,ye_sub,af] = edge_subpixel(I,xe,ye,segend)

% Finds edge locations with subpixel resolution given the original image I
% and locations of edges (with pixel resolution), xe and ye. 'segend'
% contains indices of points at ends of segments

np = length(xe);
xe_sub = zeros(size(xe)); ye_sub = zeros(size(ye));
% Adjusted flag - whether or not position was adjusted by subpix routine
af = zeros(size(xe));

% Get gradient magnitude matrix - derived from Canny filtering routine
Ig = get_grads(I);

% Transpose so that first index corresponds to x
Ig = Ig';

for ii = 1:np

  if ii == 1 | ~isempty(find(segend+1==ii))
    %use forward differencing
    seg_dirn = get_seg_dirn(xe(ii),ye(ii),xe(ii+1),ye(ii+1));
  elseif ~isempty(find(segend==ii))
    % use backward differencing
    seg_dirn = get_seg_dirn(xe(ii-1),ye(ii-1),xe(ii),ye(ii));
  else
    % use central differencing
    seg_dirn = get_seg_dirn(xe(ii-1),ye(ii-1),xe(ii+1),ye(ii+1));
  end
  
  grad_dirn = get_grad_dirn(seg_dirn);
  
  [xsub,ysub,adj] = find_grad_max(xe(ii),ye(ii),Ig,grad_dirn,0);
  
  % Old routine that used central difference approximations to the gradient
  % [xsub,ysub,adj] = calc_edge_pos_sub(xe(ii),ye(ii),I,grad_dirn);
  
  xe_sub(ii) = xsub;
  ye_sub(ii) = ysub;
  af(ii) = adj;
  
end
