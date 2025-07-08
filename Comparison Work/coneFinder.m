function [slope, intersept] = coneFinder(Img,xs,ys)
%
%
%
%
%
% Code taken from find_t9_cone by Dr. Laurence and slightly modified
global smooth_sigma trim
warning('off','all')
base_dir = pwd;
addpath([base_dir '/edge_detection'],[base_dir '/edge_tracing']);

% Settings for edge-detection
edge_type = 'canny'; %mod_edge = 1;
% Threshold for Canny detection and image smoothing factor
threshold = [0.01,0.1]; smooth_sigma = 2.0;
trim = 4;

% number of segments for top, bottom, left, right
% tsegs = 1; lsegs = 1; direct = 'ee';
orient = 'ccw'; max_dist = length(Img);
[E,~] = edge_mod(Img,edge_type,threshold,smooth_sigma);%thresh
E = E'; % make x the first index

x_inp = 0;
y_inp = 0;

% initial point
for ii = 1:length(xs)
    if E(floor(xs(ii)),floor(ys(ii))) == 1
        x_inp = floor(xs(ii));
        y_inp = floor(ys(ii));
        break
    end
end

if x_inp ~= 0;
    if x_inp > length(Img)/2
        direct = 'ww';
    else
        direct = 'ee';
    end

    % first do top
    xc = []; yc = []; %trim_ind = [];

    % Find edge point closest to inputted point x or y direction
    px = find(E(:,y_inp));
    py = find(E(x_inp,:));
    [valx,indx] = min(abs(px-x_inp));
    [valy,indy] = min(abs(py-y_inp));
    % if there are multiple closest points take the first one

    if max(size(valx)) > 1
        valx = valx(1);
        indx = indx(1);
    elseif max(size(valy)) > 1;
        valy = valy(1);
        indy = indy(1);
    end

    % Make closest point the starting point for edge tracing
    if valx <= valy
        x_start = px(indx);
        y_start = y_inp;
    else
        x_start = x_inp;
        y_start = py(indy);
    end

    % perform pixel-resolution edge detection
    [xc,yc] =  trace_edge_maxdist_trim(E,x_start,y_start,direct,orient,...
    max_dist,trim);
    segs_ind = length(xc);
    if max(size(xc)) > 1
        % perform subpixel resolution detection
        [xs,ys,~] = edge_subpixel_circ(Img,xc,yc);%af
        [xsi,ysi,~] = edge_subpixel_improved(Img,xc,yc,xs,ys,segs_ind);%af
        % remove end points
        if ~isempty(xsi)
            xsi([1,end]) = []; ysi([1,end]) = [];
        end

        % fot linear polynomial
        pp = polyfit(xsi,ysi,1);
        slope = pp(1);
        intersept = pp(2);
    else
        slope = 0;
        intersept = 0;
    end
else
    slope = 0;
    intersept = 0;
end
rmpath([base_dir '\edge_detection'],[base_dir '\edge_tracing']);
end