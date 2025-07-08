function [dir2,orient2] = get_reverse_dirn(direction,orient)

% Returns the reverse direction and orientation to those entered

if strcmp(direction,'ww')
  dir2 = 'ee';
elseif strcmp(direction,'ee')
  dir2 = 'ww';
elseif strcmp(direction,'nn')
  dir2 = 'ss';
elseif strcmp(direction,'ss')
  dir2 = 'nn';
elseif strcmp(direction,'se')
  dir2 = 'nw';
elseif strcmp(direction,'ne')
  dir2 = 'sw';
elseif strcmp(direction,'sw')
  dir2 = 'ne';
elseif strcmp(direction,'nw')
  dir2 = 'se';
end

if strcmp(orient,'cw')
  orient2 = 'ccw';
else
  orient2 = 'cw';
end