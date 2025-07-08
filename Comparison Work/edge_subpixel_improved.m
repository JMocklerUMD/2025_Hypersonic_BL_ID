function [xe_sub,ye_sub,af] = edge_subpixel_improved(I,xe,ye,xs,ys,segend)

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

ind_not_adj = [];

for ii = 1:np

  if ii == 1 | ~isempty(find(ii==segend+1))
    %use forward differencing
    seg_dirn = get_seg_dirn(xs(ii),ys(ii),xs(ii+1),ys(ii+1));
    seg_angle = atan2(ys(ii)-ys(ii+1),xs(ii+1)-xs(ii));
    % want to remove this point from fit
    ind_not_adj = [ind_not_adj,ii];
  elseif ~isempty(find(ii==segend))
    % use backward differencing
    seg_dirn = get_seg_dirn(xs(ii-1),ys(ii-1),xs(ii),ys(ii));
    seg_angle = atan2(ys(ii-1)-ys(ii),xs(ii)-xs(ii-1));
    % this point too
    ind_not_adj = [ind_not_adj,ii];
  else
    % use central differencing
    seg_dirn = get_seg_dirn(xs(ii-1),ys(ii-1),xs(ii+1),ys(ii+1));
    seg_angle = atan2(ys(ii-1)-ys(ii+1),xs(ii+1)-xs(ii-1));
  end
  
  grad_dirn = get_grad_dirn(seg_dirn);
  
  [xsub,ysub,adj] = find_grad_max(xe(ii),ye(ii),Ig,grad_dirn,1);
  %[xsub,ysub,adj] = ...
  %  find_grad_max_improved(xe(ii),ye(ii),Ig,grad_dirn,seg_angle,1);
  if adj == 0
    ind_not_adj = [ind_not_adj,ii];
  end
  % Old routine that used central difference approximations to the gradient
  % [xsub,ysub,adj] = calc_edge_pos_sub(xe(ii),ye(ii),I,grad_dirn);
 
  xe_sub(ii) = xsub;
  ye_sub(ii) = ysub;
  af(ii) = adj;
  
end

xe_sub(ind_not_adj) = [];
ye_sub(ind_not_adj) = [];

