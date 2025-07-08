function [xsub,ysub,adjusted] = find_grad_max(xc,yc,Ig,grad_dirn,mod_diag);

% Calculate direction increments
if strcmp(grad_dirn,'nn')
  dx = 0; dy = -1;
elseif strcmp(grad_dirn,'ne')
  dx = 1; dy = -1;
elseif strcmp(grad_dirn,'ee')
  dx = 1; dy = 0;
else    % 'se'
  dx = 1; dy = 1;
end

xm = xc - dx; xp = xc + dx;
ym = yc - dy; yp = yc + dy;

dIc = Ig(xc,yc);  % center point intensity
dIp = Ig(xp,yp);  % plus point (in grad_dirn)
dIm = Ig(xm,ym);  % minus point

% if abs(dIc) < abs(dIm) | abs(dIc) < abs(dIp)
%   disp(sprintf('Warning - found edge is not a maximum: gradients (%f,%f,%f); gradient direction %s\n',...
%     dIm,dIc,dIp,grad_dirn))
% end
  
% Check if all intensity gradients are equal; otherwise calculate location
% of maximum
if dIp - 2*dIc + dIm == 0 | dIc < dIm | dIc < dIp
  xsub = xc;
  ysub = yc;
  adjusted = 0;
  %disp(sprintf('Edge not maximum, mod_diag = %d',mod_diag))
else
  % normalized distance to maximum from (xc,yc) i.e. (s_0 - s_c)/ds
  ds = 0.5*(dIm - dIp)/(dIp - 2*dIc + dIm);
  
  if mod_diag == 1
    % correct if angle = 45 degrees
    if strcmp(grad_dirn,'ne') | strcmp(grad_dirn,'se')
      max_scale_fact = 0.7;
      ds_mod = sign(ds)*(0.5 - 2*(abs(ds)-0.5).^2);
      scale_fact = (dIc - 0.5*(dIm + dIp))/dIc;
      scale_fact = scale_fact/max_scale_fact;
      ds = scale_fact*ds_mod + (1 - scale_fact)*ds;
      %ds = ds_mod;
    end
  end

  xsub = xc + ds*dx;
  ysub = yc + ds*dy;
  
  adjusted = 1;
end