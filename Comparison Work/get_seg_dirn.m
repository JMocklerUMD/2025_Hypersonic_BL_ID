function dirn = get_seg_dirn(x1,y1,x2,y2)

% Returns the direction of the segment for (x1,y1) and (x2,y2) points
% Uses 3-point stencil and rounds to nearest multiple of 45 degrees

theta = atan2(y1-y2,x2-x1);

if abs(theta) <= pi/8
  dirn = 'ee';
elseif abs(theta) <= 3*pi/8
  if theta > 0
    dirn = 'ne';
  else
    dirn = 'se';
  end
elseif abs(theta) <= 5*pi/8
  if theta > 0
    dirn = 'nn';
  else
    dirn = 'ss';
  end
elseif abs(theta) <= 7*pi/8
  if theta > 0
    dirn = 'nw';
  else
    dirn = 'sw';
  end
else
  dirn = 'ww';
end
  

% Old version that used forward and backward differencing
% if x2 == x1   % north or south
%   if y2 < y1
%     dirn = 'nn';
%   else
%     dirn = 'ss';
%   end
% elseif y2 == y1   % east or west
%   if x2 > x1
%     dirn = 'ee';
%   else
%     dirn = 'ww';
%   end
% elseif x2 > x1 % easterly
%   if y2 < y1
%     dirn = 'ne';
%   else
%     dirn = 'se';
%   end
% else      % westerly
%   if y2 < y1
%     dirn = 'nw';
%   else
%     dirn = 'sw';
%   end
% end