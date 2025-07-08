function [shot, start, stop, re_ft, res, vert_reflect, horz_reflect,...
    reference_sub, gamma_adjust, pre_rot_row_crop, pre_rot_col_crop, ...
    rotate, fixed_rotation, post_rot_row_crop, post_rot_col_crop, ...
    auto_cone_crop, auto_rot_crop, auto_shadow_crop, bl_method, bl_adjust, bl_invert, peak_range,...
    power_thresh, turb_method, turb_thresh, img_dir, file_format, file_name] = read_par(shot)
%function to read the parameter file for a given shot in order to determine
%various processing details


addpath('F:\wave_packet\wave_packet_detection')
addpath('F:\wave_packet')
addpath('F:\wave_packet\parfiles_Langley')

%open the appropriate parameter file
Langley_shots = [2:81];
if sum(Langley_shots == str2num(shot)) > 0
    parfile = sprintf('C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\Comparison Work\\parfiles_Langley\\%s.par', shot); %Adjusted to use Langley parfiles
else
    parfile = sprintf('F:\\wave_packet\\parfiles\\%s.par', shot);
end
if exist(parfile, 'file')
    INFILE = fopen(parfile);
else
    out_string = sprintf('Input parameter file \''%s\'' does not exist.\n', parfile);
    error(out_string);
end

%read the text in the file
file_contents = textscan(INFILE,'%s %s', 'Delimiter','=');


%assign values to the appropriate variables
vars = matlab.lang.makeValidName(file_contents{:,1});


for i = 1:size(vars,1)
    eval([vars{i} '= cell2mat(file_contents{1, 2}(i));']);

    %convert to logicals if par gives yes or no
    if (regexp(eval(vars{i}), 'yes'))
        eval([vars{i} '= ~0;']);
        continue;
    elseif (regexp(eval(vars{i}), 'no'))
        eval([vars{i} '= ~1;']);
        continue;
    end
    
    
    %convert from string to double if applicable
    num_regex = '^\d*\.?\d*$';
    %if numel(find(isstrprop(eval(vars{i}), 'digit')==0))==0
    if regexp(eval(vars{i}), num_regex);   
        eval([vars{i} '=' eval(vars{i}) ';']);
    end
end

fclose(INFILE);

%make sure all of the return values are provided in the parameter file
%add to this list any new keywords
out_vars = {'shot', 'start', 'stop', 're_ft', 'res', 'vert_reflect', 'horz_reflect',...
    'reference_sub', 'gamma_adjust', 'pre_rot_row_crop', 'pre_rot_col_crop', ...
    'rotate', 'fixed_rotation', 'post_rot_row_crop', 'post_rot_col_crop', ...
    'auto_cone_crop', 'auto_rot_crop', 'auto_shadow_crop', 'bl_method', 'bl_adjust', 'bl_invert', 'peak_range',...
    'power_thresh', 'turb_method', 'turb_thresh', 'img_dir', 'file_format', 'file_name'};
    
missing = {};
for curr_var = out_vars
    if ~ exist(cell2mat(curr_var), 'var')
        missing{end+1} = curr_var;
    end
end
    
%throw error if any essential values are not returned
if numel(missing) > 0
    out_string = sprintf('Invalid parameter file. The following fields are missing:\n\n');
    for curr_var = missing
        out_string = [out_string sprintf('\t %s \n', cell2mat(curr_var{1,1}))];
    end
    error(out_string);
end


end

