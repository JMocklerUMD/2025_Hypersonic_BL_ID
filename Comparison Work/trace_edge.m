function [x_seg,y_seg] = ...
    trace_edge(I,x_start,y_start,start_dir,orientation)

% Traces the points on an edge in the given orientation
% NB: has been modified for modified Canny detector

global trim

% Default direction is counter-clockwise
if nargin == 4
	orientation = 'ccw';
end

% 2*wlength+1 is length of window for comparing angles for line segments on
% either side of wlength edge points behind the current one. If the difference
% in angle is greater than max_dev in wrong direction (i.e. that with wrong
% curvature for current orientation)
% Values of 7 and 25 (degrees) seem to work well
wlength = 7;
max_dev = 25;  % in degrees  

% Initialise segment vectors
x_seg(1) = x_start;
y_seg(1) = y_start;
x_old = x_start;
y_old = y_start;

% Setup order of directions for current orientation
if strcmp(orientation,'cw')
  arrow = {'nn','ne','ee','se','ss','sw','ww','nw'};
else
	arrow = {'nn','nw','ww','sw','ss','se','ee','ne'};
end

% Look for edge in starting direction. If not there, look in directions
%		on either side. If can't find, return error.
% Also sets the current direction and previous direction to look for new
% edges
start_ind = find(strcmp(arrow,start_dir));
edge_found = 0;
[x_new,y_new] = get_new_indices(start_dir,x_start,y_start);

if I(x_new,y_new) == 1  % edge point in start direction found
	edge_found = 1;
	current_dir = start_dir;
  % Set the last direction to be the previous value in 'arrow'
  last_dir = arrow{mod(start_ind-2,8)+1};
else
  % Try next direction as alternative
  alt_start_ind = mod(start_ind,8) + 1;
	[x_new,y_new] = get_new_indices(arrow{alt_start_ind},x_start,y_start);

	if I(x_new,y_new) == 1  % edge point in alternative direction found
		edge_found = 1;
		last_dir = start_dir;
		current_dir = arrow{alt_start_ind};
  else
    % Try previous direction as last alternative
    alt_start_ind = mod(start_ind-2,8) + 1;
    [x_new,y_new] = ...
        get_new_indices(arrow{alt_start_ind},x_start,y_start);

    if I(x_new,y_new) == 1  % edge point in this direction found
			edge_found = 1;
			last_dir = start_dir;
			current_dir = arrow{alt_start_ind};
		end
	end
end

% Bad choice of start direction if no edge found -> return error
if edge_found == 0
	herr = errordlg('No edge found in specified direction')
    waitfor(herr)
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start edge tracing - find all points in current segment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start from second point and find all points on segment
n = 2;
current_dir_ind = find(strcmp(arrow,current_dir));

% Setup x and y window vectors of length 2*wlength+1, which contain points
%   up to and including the current point. The angle between the lines 
%   defined by the first and middle points and the last and middle points 
%   are compared, and if the exceeds the angle max_dev in the wrong
%   (convex) direction for the current orientation, the routine finishes. 
xw = zeros(1,2*wlength+1);
yw = zeros(1,2*wlength+1);

% Convert to radians
max_dev = max_dev*pi/180;

finished = 0;
max_dev_exceeded = 0;
while finished == 0
    
    % Update segment vectors
    % Check that diagonals haven't skipped pixels on either side
    %dist_sq = (x_new-x_old)^2 + (y_new-y_old)^2
    if (x_new-x_old)^2 + (y_new-y_old)^2 == 2
      
      if I(x_new,y_old) == 1
        x_seg(n) = x_new;
        y_seg(n) = y_old;
        x_seg(n+1) = x_new;
        y_seg(n+1) = y_new;
        n = n+2;
      elseif I(x_old,y_new) == 1
        x_seg(n) = x_old;
        y_seg(n) = y_new;
        x_seg(n+1) = x_new;
        y_seg(n+1) = y_new;
        n = n+2;
      else
        x_seg(n) = x_new;
        y_seg(n) = y_new;
        n = n+1;
      end
    else
      x_seg(n) = x_new;
      y_seg(n) = y_new;
      n = n+1;
    end
    
    x_old = x_new;
    y_old = y_new;
    
    % Indices in 'arrow' of possible directions for the next point in 
    %   segment, in order of priority
    poss_dirs_ind = [current_dir_ind, current_dir_ind+1,...
            current_dir_ind+2, current_dir_ind-1, current_dir_ind-2];
    poss_dirs_ind = mod(poss_dirs_ind-1,8) + 1;
    no_dirs = length(poss_dirs_ind);

    % Check if a terminal point (only one neighbour), if so try skipping 
    %   a point in current direction - no longer used as found to be
    %   unreliable
%     no_neighbs = length(find(I(x_old-1:x_old+1,y_old-1:y_old+1))) - 1;
%     if no_neighbs == 1
%         [x_old,y_old] = ...
%             get_new_indices(arrow{poss_dirs_ind(1)},x_old,y_old);
%     end
    
    % Look for next point in segment
    finished = 1;
    for jj = 1:no_dirs
        [x_new,y_new] = ...
            get_new_indices(arrow{poss_dirs_ind(jj)},x_old,y_old);
        if I(x_new,y_new) == 1
            current_dir_ind = poss_dirs_ind(jj);
            
            % Update window vectors for comparing angles
            for kk = 1:2*wlength
                xw(kk) = xw(kk+1);
                yw(kk) = yw(kk+1);
            end
            xw(2*wlength+1) = x_new;
            yw(2*wlength+1) = y_new;
            
            % Check that max_dev angle is not exceeded
            if length(find(xw)) == 2*wlength+1
                % Angle between current and current-wlength points ...
                alphaa = atan2(yw(wlength+1)-yw(2*wlength+1),...
                    xw(2*wlength+1)-xw(wlength+1));
                % and between current-wlength and current-2*wlength points
                alphab = atan2(yw(1)-yw(wlength+1),...
                    xw(wlength+1)-xw(1));
                
                % Check difference, adjusting if gone through -pi/pi
                alpha_diff = alphaa - alphab;
                if strcmp(orientation,'ccw')
                    if alpha_diff < -3*pi/2
                        alpha_diff = 2*pi + alpha_diff;
                    end
                    if alpha_diff < -max_dev
                      max_dev_exceeded = 1;
                      break
                    end
                else
                    if alpha_diff > 3*pi/2
                        alpha_diff = alpha_diff - 2*pi;
                    end
                    if alpha_diff > max_dev
                      max_dev_exceeded = 1;
                      break
                    end
                end
            end
            finished = 0;
            break
        end
    end
     
end

% Last few values could be suspect, especially if the max. deviation angle
% was exceeded
%if max_dev_exceeded == 1
if 0
  x_seg = x_seg(1:length(x_seg)-wlength-trim);
  y_seg = y_seg(1:length(y_seg)-wlength-trim);
else
  x_seg = x_seg(1:length(x_seg)-trim);
  y_seg = y_seg(1:length(y_seg)-trim);
end
