function [x_seg,y_seg] = ...
    trace_edge_circ(I,x_start,y_start,start_dir,orientation,trim)

% Traces the points on an edge in the given orientation

% Default direction is counter-clockwise
if nargin == 4
	orientation = 'ccw';
end

% Store starting values so that doesn't get stuck in endless loop
x0 = x_start;
y0 = y_start;

%trim = 7;       % Number of points to trim off end of segment vectors
wlength = 5;    % 2*wlength+1 is length of window for comparing angles
max_dev = 20;   % Maximum deviation in wrong direction between before 
                %   and after angles 

% Initialise segment vectors
x_seg(1) = x_start;
y_seg(1) = y_start;

% Setup directions
if strcmp(orientation,'cw')
    arrow = {'nn','ne','ee','se','ss','sw','ww','nw'};
else
	arrow = {'nn','nw','ww','sw','ss','se','ee','ne'};
end

% Look for edge in starting direction. If not there, look in directions
%		on either side. If can't find, return error.
start_ind = find(strcmp(arrow,start_dir));

edge_found = 0;
[x_new,y_new] = get_new_indices(start_dir,x_start,y_start);

if I(x_new,y_new) == 1  % edge point in start direction found
	edge_found = 1;
	current_dir = start_dir;
    % Assume a last direction
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

% Bad choice of start direction if no edge found
if edge_found == 0
	herr = errordlg('No edge found in specified direction')
    waitfor(herr)
    return
end

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
while finished == 0
    
    % Update segment vectors
    x_seg(n) = x_new;
    y_seg(n) = y_new;
    n = n+1;
    
    x_old = x_new;
    y_old = y_new;
    
    if x_old == x0 & y_old == y0
      finished = 1;
      break
    end
    
    % Indices in 'arrow' of possible directions for the next point in 
    %   segment, in order of priority
    poss_dirs_ind = [current_dir_ind, current_dir_ind+1,...
            current_dir_ind+2, current_dir_ind-1, current_dir_ind-2];
    poss_dirs_ind = mod(poss_dirs_ind-1,8) + 1;
    no_dirs = length(poss_dirs_ind);

    % Check if a terminal point (only one neighbour), if so try skipping 
    %   a point in current direction
    no_neighbs = length(find(I(x_old-1:x_old+1,y_old-1:y_old+1))) - 1;
    if no_neighbs == 1
%         [x_old,y_old] = ...
%             get_new_indices(arrow{poss_dirs_ind(1)},x_old,y_old);
    end
    
    % Look for next point in segment
    finished = 1;
    for jj = 1:no_dirs
        [x_new,y_new] = ...
            get_new_indices(arrow{poss_dirs_ind(jj)},x_old,y_old);
        if I(x_new,y_new) == 1
            current_dir_ind = poss_dirs_ind(jj);
            
            % Update window vectors
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
                        break
                    end
                else
                    if alpha_diff > 3*pi/2
                        alpha_diff = alpha_diff - 2*pi;
                    end
                    if alpha_diff > max_dev
                        break
                    end
                end
            end
            finished = 0;
            break
        end
    end
    
    
    
end

% Last few values could be suspect
x_seg = x_seg(1:length(x_seg)-trim);
y_seg = y_seg(1:length(y_seg)-trim);
