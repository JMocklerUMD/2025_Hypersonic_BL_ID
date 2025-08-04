folder_path = 'C:\Users\cathe\Documents\MATLAB\trainML';
file_path = fullfile(folder_path, 'Langley_Run34_train.txt');
lines = readlines(file_path);
len_lines = length(lines);

WL = [];
k =0;

for i = 1:10%len_lines-1
    dat = strtrim(lines(i));
    results = strsplit(dat, '\t');
    wp_id = results{2};
    rows = str2double(results{7});
    cols = str2double(results{8});
    img_data = str2double(results(9:end));
    img = reshape(img_data, cols, rows)';
    [rows_crop, cols_crop] = size(img);
    img = imcrop(img, [0, 0, 1200, 55]);

    if wp_id == '1'
        k = k+1;
        WL(k) = findwavelength(17,img);   
    end
end

WL_avg = mean(WL)

%%


function WL = findwavelength(BL_height, img)
    [rows, cols] = size(img);
    top_BL = rows- BL_height;
    X = img(top_BL-2,:);
    
    Fs = 1;            % Sampling frequency                    
    T = 1/Fs;             % Sampling period       
    L = 784;             % Length of signal
    t = (0:L-1)*T;        % Time vector
    
    Y = fft(X);
    
    WL_min = 2*BL_height;
    WL_max = 3*BL_height;
    
    f = Fs*(0:L/2)/L;
    
    Y_mag = abs(Y(1:L/2+1));
    Y_mag(2:end-1) = 2 * Y_mag(2:end-1);
    
    low = find(f >= 1/WL_max, 1, 'first');
    up = find(f <= 1/WL_min, 1, 'last');
    
    [~, freq_idx_rel] = max(Y_mag(low:up));
    freq_idx = low + freq_idx_rel - 1;
    
    freq = f(freq_idx);
    WL = 1/freq;

    figure
    plot(f, Y_mag, 'LineWidth', 1)
    hold on
    plot(freq, Y_mag(freq_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 1)  % Peak point
    title("Complex Magnitude of fft Spectrum with Peak")
    xlabel("f (Hz)")
    ylabel("|fft(X)|")
    legend("FFT Magnitude", "Peak Frequency")
    
end



%%
% Plot FFT with peak point
figure
plot(f, Y_mag, 'LineWidth', 1)
hold on
plot(freq, Y_mag(freq_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 1)  % Peak point
title("Complex Magnitude of fft Spectrum with Peak")
xlabel("f (Hz)")
ylabel("|fft(X)|")
legend("FFT Magnitude", "Peak Frequency")




