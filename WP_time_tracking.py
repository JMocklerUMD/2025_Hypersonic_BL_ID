# -*- coding: utf-8 -*-
"""
DATE:           Wed Jun 25 09:36:27 2025
AUTHOR:         Joseph Mockler
DESCRPITION:    This script takes raw classification results and first filters 
                them for coherence in both time and space. Then, the script 
                reconstruct the signal between frames and performs a correlation
                -based analysis to compute the 2nd mode wave packet convection
                speed inside the boundary layer. Method works best in 1D, but
                can also work in 2D (with both scripts provided)
"""

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import os
import math

#%% Load classification results + video
Confidence_history = np.load('C:\\UMD GRADUATE\\RESEARCH\\Hypersonic Image ID\\videos\\Test1\\classification_results_run38_filtered.npy')

print('Reading training data file')

# Write File Name
file_name = 'C:\\UMD GRADUATE\\RESEARCH\\Hypersonic Image ID\\videos\\Test1\\run38\\video_data_100_109ms_Run33.txt'
if os.path.exists(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len = len(lines)
print(f"{lines_len} lines read")


#%% Fnc calls

# Split the image into 20 pieces
def image_splitting(i, lines):
    WP_io = []
    #SM_bounds_Array = []
    Imagelist = []
    
    curr_line = i;
    line = lines[curr_line]
    
    parts = line.strip().split()
    
    run = parts[0]
    image_response = parts[1]
    sm_check = parts[2]
    if sm_check.startswith('X'):
    	sm_bounds = list(map(str, parts[2:6]))  # Convert bounds to integers
    else:
    	sm_bounds = list(map(int, parts[2:6]))
    image_size = list(map(int, parts[6:8]))  # Convert image size to integers
    image_data = list(map(float, parts[8:]))  # Convert image data to floats
    
    # Reshape the image data into the specified image size
    full_image = np.array(image_data).astype(np.float64)
    full_image = full_image.reshape(image_size)  # Reshape to (rows, columns)
    
    #if full_image.shape != (64, 1280):
    #    print(f"Skipping image at line {i+1} — unexpected size {full_image.shape}")
        #continue
    
    slice_width = 64
    height, width = full_image.shape
    num_slices = width // slice_width
    
    # Only convert bounds to int if not sm_check.startswith('X')
    if not sm_check.startswith('X'):
        sm_bounds = list(map(int, sm_bounds))
        x_min, y_min, box_width, box_height = sm_bounds
        x_max = x_min + box_width
        y_max = y_min + box_height
    
    for i in range(num_slices-1):
        x_start = i * slice_width
        x_end = (i + 1) * slice_width
    
        # Slice the image
        image = full_image[:, x_start:x_end]
        image_size = image.shape
        Imagelist.append(image)
    
        if sm_check.startswith('X'):
            WP_io.append(0)
    
        else:
            # Check for horizontal overlap with this slice
            if x_max >= x_start+slice_width/4 and x_min <= x_end-slice_width/4:
                WP_io.append(1)
    
            else:
                WP_io.append(0)
                
    return Imagelist, WP_io, slice_width, height, sm_bounds


def calc_windowed_confid(j, confidence, window_size):
    '''
    Calculates the local confidence (i.e. a single slice of a frame) 
    via a summed windowing method
    '''
    if (j - window_size//2) < 0: # at the front end of the image
        local_confid = np.sum(confidence[0:j+window_size//2+1:1])
    elif (j + window_size//2) > len(confidence): # at the end of the image list
        local_confid = np.sum(confidence[j-window_size//2-1:len(confidence):1])
    else:
        local_confid = np.sum(confidence[j-window_size//2:j+window_size//2+1:1])
        
    return local_confid

def classify_the_frame(Imagelist, confidence, WP_io, window_size, indiv_thres, confid_thres, use_post_process):
    n00, n01, n10, n11 = 0, 0, 0, 0 
    filtered_result = []
    classification_result = np.zeros(len(Imagelist))
    for i, _ in enumerate(Imagelist):
        
        # If using the windowed post processing, call the windowing fcn
        # to get the locally informed confidence. Then compare to thresholds
        if use_post_process == 1:
            local_confid = calc_windowed_confid(i, confidence, window_size)
            
            # Are window and indiv conditions met?
            if (local_confid > confid_thres) or (confidence[i] > indiv_thres):
                filtered_result.append(1)
            else:
                filtered_result.append(0)
            
            classification_result[i] = filtered_result[i]
        
        # If not, then just round
        else:
            classification_result[i] = np.round(confidence[i])
            
        # Get stats on the current image
        if WP_io[i] == 0:
            if classification_result[i] == 0:
                n00 += 1
            if classification_result[i] == 1:
                n01 += 1 
        elif WP_io[i] == 1:
            if classification_result[i] == 0:
                n10 += 1
            if classification_result[i] == 1:
                n11 += 1
                
    return classification_result, filtered_result, n00, n01, n10, n11

#%% Filter the ID's in time and space

window_size = 3
confid_thres = 1.5
indiv_thres = 0.9
use_post_process = 1
WP_locs_list = []

for i in range(2495):
    # Start by classifying i'th frame
    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i, lines)
    confidence = Confidence_history[i]
    
    # Filtered result includes the filtering in space
    classification_result, filtered_result, n00, n01, n10, n11 = classify_the_frame(Imagelist, confidence, WP_io, window_size, indiv_thres, confid_thres, use_post_process)
    
    # Perform a "lookahead" of the next 3 frames
    # Sum the classifications along the time dimension 
    # (e.g form a 19x1 array that's the sum 4 time steps of classification results)
    WP_locs = np.zeros(len(Imagelist))
    WP_locs = WP_locs + filtered_result
    for ii in range(1,4):
        Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i + ii, lines)
        confidence = Confidence_history[i+ii]
        classification_result, filtered_result, n00, n01, n10, n11 = classify_the_frame(Imagelist, confidence, WP_io, window_size, indiv_thres, confid_thres, use_post_process)
        
        # Sum in time
        WP_locs = WP_locs + filtered_result
        
    # Now go thru the list and ID where, among the 4 frames, there's consistency
    # NOTE: change the confidence level below (WP_candidate > 3) to a higher or 
    # for a different threshold
    start_slice = 0
    stop_slice = 0
    consec = 0
    for j, WP_candidate in enumerate(WP_locs):
        # Determine when we first detect a WP moving left-right
        if WP_candidate > 3 and consec == 0:
            start_slice = j
            consec = 1
        
        # Now see when the WP stops
        if WP_candidate < 3 and consec == 1:
            consec = 0
            stop_slice = j
            
            # Perform a correction to account for left-right convection
            if WP_locs[start_slice-1] > 0:
                start_slice = start_slice - 1
                
            # Handle when the WP is at the start of the frame
            if start_slice < 0:
                start_slice = 0
                WP_locs_list.append([start_slice, stop_slice])
            break # break the enumerate(WP_locs) for loop
    
        # Handle advection off the screen
        if consec == 1 and j == len(WP_locs) - 1:
            stop_slice = j
    
    # For a list of start-stop slices for the entire analyzed video set
    WP_locs_list.append([start_slice, stop_slice])        
    
    # Print and inspect if you want
    #print(f"Frame {i}: WP_loc = [{start_slice}, {stop_slice}]")          
    
    
#%% Perform 1D propagation analysis via correlation - necessary fcns

def subpixel_convection_speed(lags, corr, window_polyfit, start_lag, stop_lag):
    
    # Find the max points
    start_idx = np.where(lags == start_lag)[0][0]
    end_idx = np.where(lags == stop_lag)[0][0]
    corr_idx = np.argmax(corr[start_idx:end_idx]) + start_idx # Note that corrmax and lagmax share the same idx
    # corr_idx = np.argmax(corr[start_lag:stop_lag]) + start_lag
    
    # Pick range to fit the 2nd order curve over
    lag_chopped = lags[corr_idx-window_polyfit:corr_idx+window_polyfit]
    corr_chopped = corr[corr_idx-window_polyfit:corr_idx+window_polyfit]
    
    # Perform the fit and find the maximum
    pfit = np.polyfit(lag_chopped, corr_chopped, 2)
    subpix_convect = pfit[1]/(2*pfit[0])
    
    return subpix_convect

def correlate_1D_signals(signal1, signal2):
    lags = signal.correlation_lags(len(signal1), len(signal2))
    corr = signal.correlate(signal1, signal2)
    return lags, corr


def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2_matlab(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r


def plot_reconstructed_signals(line1, recreated_signals, line2):
    line125 = recreated_signals[0,:]
    line150 = recreated_signals[1,:]
    line175 = recreated_signals[2,:]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize = (15,12))
    ax1.plot(range(len(line1)), line1)
    ax1.plot(range(len(line125)), line125)
    ax1.set_ylim(-0.25, 0.25)
    ax2.plot(range(len(line125)), line125)
    ax2.plot(range(len(line150)), line150)
    ax2.set_ylim(-0.25, 0.25)
    ax3.plot(range(len(line150)), line150)
    ax3.plot(range(len(line175)), line175)
    ax3.set_ylim(-0.25, 0.25)
    ax4.plot(range(len(line175)), line175)
    ax4.plot(range(len(line2)), line2)
    ax4.set_ylim(-0.25, 0.25)
    plt.show()
    
def plot_analyzed_frames(imageReconstruct1, row_search, imageReconstruct2):
    # Plot the image
    fig, (ax_orig, ax_template) = plt.subplots(2, 1, figsize=(15, 6))
    ax_orig.imshow(imageReconstruct1, cmap='gray')
    ax_orig.set_title('Frame 1')
    ax_orig.axhline(y=row_search, color='r', linestyle='-')
    ax_orig.set_axis_off()
    ax_template.imshow(imageReconstruct2, cmap='gray')
    ax_template.set_title('Frame 2')
    ax_template.axhline(y=row_search, color='r', linestyle='-')
    ax_template.set_axis_off()
    fig.show()
    
#%% Perform 1D propagation analysis via correlation

# First calculate approx how many pixels a wave will propogate in a single frame
mm_pix = 0.0756         # From paper
FR = 285e3              # Camera frame rate in Hz
dt = 1/FR               # time step between frames
prop_speed_low = 500                # Lower cutoff for assumed prop speed, m/s
prop_speed_high = 1250              # Upper cutoff for assumed prop speed, m/s
row_search = 38                     # Row to take measurements through

pix_tr_low = prop_speed_low * dt * 1/(mm_pix*1e-3)  # Computed number of pixels traveled between frames
pix_tr_high = prop_speed_high * dt * 1/(mm_pix*1e-3)

# Processing parameters
use_bandpass_filter = 1      # Use the BP filter - recommended! But set to 0 if already pre-filtered
window_polyfit = 5      # Window for computing convection speed from polyfit
signal_buffer = 2       # How much extra "slice" should be kept outside of the localized WP?
corr_max_search = 10    # What range should you search for the max
start_lag = -round(pix_tr_high)     # Negative is taken for "backwards" correlation correction
stop_lag = -round(pix_tr_low)

# Bandpass filter the frames
sos = signal.butter(4, [0.0133, 0.080], btype='bandpass', fs=1, output='sos')

convect_speed = []
convect_breakout = []
for ii in range(2495):
    
    # Pick which WP frames we should analyze
    if ii == 0:
        convect_breakout.append([])
        continue
    
    # Check that the WP ii and ii+1 are larg enough to get good results from
    if WP_locs_list[ii][1] - WP_locs_list[ii][0] < 3 or WP_locs_list[ii+1][1] - WP_locs_list[ii+1][0] < 3:
        convect_breakout.append([])
        continue
    
    # Extract out the starting and stopping values
    WP_loc_ii = WP_locs_list[ii]
    starting_WP = WP_loc_ii[0]
    stopping_WP = WP_loc_ii[1]
    
    start_analysis = max(starting_WP - signal_buffer, 0)
    stop_analysis = min(stopping_WP + signal_buffer, 18)
    
    # Take two frames where we know WP's exist
    Imagelist1, WP_io, slice_width, height, sm_bounds = image_splitting(ii, lines)
    Imagelist2, WP_io, slice_width, height, sm_bounds = image_splitting(ii+1, lines)
    imageReconstruct1 = np.hstack(Imagelist1[start_analysis:stop_analysis])     # WP's should be at 10 to 18
    imageReconstruct2 = np.hstack(Imagelist2[start_analysis:stop_analysis])     # WP's should be at 10 to 18
    
    # Take some slices at the boundary layer to perform the correlation
    line1 = imageReconstruct1[row_search,:]
    line2 = imageReconstruct2[row_search,:]
    
    # Apply the filter to the signal
    if use_bandpass_filter == 1:
        line1 = signal.sosfiltfilt(sos, line1)
        line2 = signal.sosfiltfilt(sos, line2)
    
    # Compute the cross-correlation
    lags, corr = correlate_1D_signals(line1, line2)
    
    # Calculate the convection speed via a correlation analysis
    subpix_convect = subpixel_convection_speed(lags, corr, window_polyfit, start_lag, stop_lag)
    convect_speed.append(subpix_convect)
    
    # Append to a master list for later analysis
    convect_breakout.append(convect_speed[-1:])

# Perform some basic statistics
mean_speed = np.mean(convect_speed)*mm_pix *1e-3 / dt
std_dev = np.std(convect_speed)*mm_pix *1e-3 / dt

print(f'Row number: {row_search}')
print(f'Mean: {mean_speed} +/- {std_dev}')
#print(convect_breakout)

# Form a histogram and inspect the results
counts, bins = np.histogram([prop*mm_pix *1e-3 / dt for prop in convect_speed], bins=20)
fig, ax1 = plt.subplots(1)
ax1.stairs(counts, bins)
ax1.set_xlabel('Measured propagation speed, m/s')
ax1.set_ylabel('Frequency counts')
ax1.set_xlim(650, 1000)
plt.show()


#%%
# Take two frames where we know WP's exist
row_search = 38
start_analysis = WP_locs_list[420][0]
stop_analysis = WP_locs_list[420][1]
Imagelist1, WP_io, slice_width, height, sm_bounds = image_splitting(420, lines)
Imagelist2, WP_io, slice_width, height, sm_bounds = image_splitting(421, lines)
imageReconstruct1 = np.hstack(Imagelist1[start_analysis:stop_analysis])     # WP's should be at 10 to 18
imageReconstruct2 = np.hstack(Imagelist2[start_analysis:stop_analysis])     # WP's should be at 10 to 18
plot_analyzed_frames(imageReconstruct1, row_search, imageReconstruct2)

#%% correlation plot thing
fig, ax = plt.subplots(1)
ax.plot(lags, corr)
ax.set_ylabel('Computed Correlation')
ax.set_xlabel('Pixel lead/lag')
ax.axvline(x=lags[np.argmax(corr)], color='r', linestyle='-')
ax.set_xlim(-100,100)
plt.show()

#%%
fig, ax = plt.subplots(1)
start_idx = np.where(lags == start_lag)[0][0]
end_idx = np.where(lags == stop_lag)[0][0]
corr_idx = np.argmax(corr[start_idx:end_idx]) + start_idx
ax.scatter(lags[corr_idx-window_polyfit:corr_idx+window_polyfit], corr[corr_idx-window_polyfit:corr_idx+window_polyfit])

# Perform the fit and find the maximum
pfit = np.polyfit(lags[corr_idx-window_polyfit:corr_idx+window_polyfit], corr[corr_idx-window_polyfit:corr_idx+window_polyfit], 2)
x = np.linspace(lags[corr_idx-window_polyfit], lags[corr_idx+window_polyfit])
y = pfit[0]*x**2 + pfit[1]*x + pfit[2]
ax.plot(x, y, 'k--')
ax.axvline(x=-pfit[1]/(2*pfit[0]), color='r', linestyle='-')
ax.set_xlabel('Pixel lead/lag')
ax.set_ylabel('Correlation coeff')
plt.show()


#%% Analyzing the results to see what's going on

# Does WP size correlate with measurement performance?
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (13,5))
size_frame = []
mean_convect_frame = []
std_convect_frame = []

for i in range(len(convect_breakout)):
    convect_measurements = convect_breakout[i]
    std_dev_meas = np.std(convect_measurements)
    mean_meas = np.mean(convect_measurements)
    locs = WP_locs_list[i]
    size = locs[1] - locs[0]
    
    size_frame.append(size)
    mean_convect_frame.append(mean_meas)
    std_convect_frame.append(std_dev_meas)
    
ax1.scatter(size_frame, mean_convect_frame)
ax1.set_xlabel('Detected WP length')
ax1.set_ylabel('Mean prop speed (pix) between 2 frames')
#ax1.set_ylim(-70, -10)
ax2.scatter(size_frame, std_convect_frame)
ax2.set_xlabel('Detected WP length')
ax2.set_ylabel('Stdev prop speed (pix) between 2 frames')
plt.show()

# What about confidence?
mean_confid = []
mean_convect_frame = []
std_convect_frame = []
for i in range(len(convect_breakout)):
    convect_measurements = convect_breakout[i]
    std_dev_meas = np.std(convect_measurements)
    mean_meas = np.mean(convect_measurements)
    locs = WP_locs_list[i]
    confid = Confidence_history[i]
    confid_list = []
    for j in range(locs[0], locs[1]+1):
        confid_list.append(confid[j])
        
    mean_confid.append(np.mean(confid_list))
    mean_convect_frame.append(mean_meas)
    std_convect_frame.append(std_dev_meas)

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (13,5))
ax1.scatter(mean_confid, mean_convect_frame)
ax1.set_xlabel('WP Classifier Confidence')
ax1.set_ylabel('Mean prop speed (pix) between 2 frames')
#ax1.set_ylim(-70, -10)
ax2.scatter(mean_confid, std_convect_frame)
ax2.set_xlabel('WP Classifier Confidence')
ax2.set_ylabel('Stdev prop speed (pix) between 2 frames')
plt.show()

#%% 2D correlation method

# First calculate approx how many pixels a wave will propogate in a single frame
mm_pix = 0.0756         # From paper
FR = 285e3              # Camera frame rate in Hz
dt = 1/FR               # time step between frames
prop_speed = 850        # A priori estimate of propogation speed

prop_speed_low = 500                # Lower cutoff for assumed prop speed, m/s
prop_speed_high = 1250              # Upper cutoff for assumed prop speed, m/s

bl_range_start = 35
bl_range_end = 55

pix_tr_low = prop_speed_low * dt * 1/(mm_pix*1e-3)  # Computed number of pixels traveled between frames
pix_tr_high = prop_speed_high * dt * 1/(mm_pix*1e-3)

# Processing parameters
use_bandpass_filter = 1 # Use the BP filter? Recommended if not already filtered
use_windowing_method = 1        # Uses the windowing approach that Dr. Laurence mentioned
window_size = 64
use_2D_corr_method = 0          # Uses the 2D correlation approach
window_polyfit = 5      # Window for computing convection speed from polyfit
signal_buffer = 0       # How much extra "slice" should be kept outside of the localized WP?
corr_max_search = 10    # What range should you search for the max
start_lag = -round(pix_tr_high)     # Negative is taken for "backwards" correlation correction
stop_lag = -round(pix_tr_low)

# Bandpass filter the frames
sos = signal.butter(4, [0.0133, 0.080], btype='bandpass', fs=1, output='sos')

convect_speed = []
convect_breakout = []
for k in range(2495):
    
    # This stuff just does filtering if we should bother analyzing this frame
    if k == 0:
        convect_breakout.append([])
        continue
    
    # Check that the WP ii and ii+1 are larg enough to get good results from
    if WP_locs_list[k][1] - WP_locs_list[k][0] < 3 or WP_locs_list[k+1][1] - WP_locs_list[k+1][0] < 3:
        convect_breakout.append([])
        continue
    
    WP_loc_ii = WP_locs_list[k]
    starting_WP = WP_loc_ii[0]
    stopping_WP = WP_loc_ii[1]
    
    start_analysis = max(starting_WP - 2, 0)
    stop_analysis = min(stopping_WP + 2, 18)
    
    # Take two frames where we know WP's exist
    Imagelist1, WP_io, slice_width, height, sm_bounds = image_splitting(k, lines)
    Imagelist2, WP_io, slice_width, height, sm_bounds = image_splitting(k+1, lines)
    imageReconstruct1 = np.hstack(Imagelist1[start_analysis:stop_analysis])     
    imageReconstruct2 = np.hstack(Imagelist2[start_analysis:stop_analysis])     
    
    # Reshape the frames for the cutoff locs we described above
    img1 = imageReconstruct1[bl_range_start:bl_range_end, :]
    img2 = imageReconstruct2[bl_range_start:bl_range_end, :]
    
    row_range, col_range = img1.shape
    
    # Bandpass filter the sliced 2D frame
    if use_bandpass_filter == 1:
        for i in range(row_range):
            img1[i,:] = signal.sosfiltfilt(sos, img1[i,:])
            img2[i,:] = signal.sosfiltfilt(sos, img2[i,:])
    
    
    # WINDOWING APPROACH: slides along the frame and correlates the reference image
    # i.e. the first window of the image, to the i+1 frame window. Then uses this recreated
    # correlation-lag plot to find the maximum
    if use_windowing_method == 1:
        # Initialize the first frame
        fr1 = img1[:, 0:window_size]
        corr2 = []
        lags2 = []
        
        # Compute the sliding window
        for j in range(col_range-window_size):
            fr2 = img2[:, j:window_size+j]
            corr_coef_j = corr2_matlab(fr1, fr2)
            corr2.append(corr_coef_j)
            lags2.append(j)
            
        # Compute the correlation index
        corr_idx = np.argmax(corr2[-stop_lag:-start_lag]) + (-stop_lag)
        
        # Pick range to fit the 2nd order curve over
        lag_chopped = lags2[corr_idx-window_polyfit:corr_idx+window_polyfit]
        corr_chopped = corr2[corr_idx-window_polyfit:corr_idx+window_polyfit]
        
        # Perform the fit and find the maximum
        pfit = np.polyfit(lag_chopped, corr_chopped, 2)
        subpix_convect = pfit[1]/(2*pfit[0])
        
        # Filter spurious results
        if abs(subpix_convect) > pix_tr_high or abs(subpix_convect) < pix_tr_low:
            continue
        
        # Save off for plotting and analysis
        convect_speed.append(subpix_convect)
           
        #print(f"Frame: {k}. Measured convection speeds {convect_speed[-(N_interps+1):]}")
        convect_breakout.append(convect_speed[-1:])
    
    # 2D CORRELATION METHOD: computes the 2D correlation between frame k and k+1 and
    # finds the maximum among the 2D array. The horizontal slice through this maximum is
    # taken and processed as a 1D slice.
    if use_2D_corr_method == 1:
        # Calculate the 2D correlation
        lags2 = signal.correlation_lags(len(img1[1,:]), len(img2[1,:]), mode = "same")
        corr2 = signal.correlate2d(img1, img2, boundary='symm', mode='same')
        
        # Determine the max point
        row_max, col_max = np.unravel_index(np.argmax(corr2), corr2.shape)
        
        # Back out the subpixel convection speed
        subpix_convect = subpixel_convection_speed(lags2, corr2[row_max,:], window_polyfit, start_lag, stop_lag)
        
        # Filter spurious results
        if abs(subpix_convect) > pix_tr_high or abs(subpix_convect) < pix_tr_low:
            continue
        
        # Save off for plotting and analysis
        convect_speed.append(subpix_convect)
           
        #print(f"Frame: {k}. Measured convection speeds {convect_speed[-(N_interps+1):]}")
        convect_breakout.append(convect_speed[-1:])


mean_speed = np.mean(convect_speed)*mm_pix *1e-3 / dt
std_dev = np.std(convect_speed)*mm_pix *1e-3 / dt

print(f'Mean: {mean_speed} +/- {std_dev}')

#%% Form a histogram and inspect the results
counts, bins = np.histogram([prop*mm_pix *1e-3 / dt for prop in convect_speed], 20)
fig, ax1 = plt.subplots(1)
ax1.stairs(counts, bins)
ax1.set_xlabel('Measured propagation speed, m/s')
ax1.set_ylabel('Frequency counts')
#ax1.set_xlim(700, 1000)
ax1.set_xlim(-1100, -600)
plt.show()

#%% Frame reconsturction - 2D
#N_interps = 0
#increment = round(pix_tr/(N_interps+1))
#cutoff_len = round(pix_tr)
# # Our recreated signals should chop off the front and back bit to avoid
# # any weird clipping I noticed
# recreated_signals = np.zeros((N_interps, row_range, col_range))

# # ii forms the intermediate frames
# for ii in range(1,N_interps+1):
    
#     # j iterates over the cols (weighted avg is equally applied across the entire col)
#     for j in range(cutoff_len, col_range-cutoff_len):
#         # Look backwards i*increment from the
#         step_bwd = ii*increment
#         step_fwd = ((N_interps+1)-ii)*increment
#         recreated_signals[ii-1,:,j] = (((N_interps+1)-ii)*imageReconstruct1[:,j-step_bwd] + ii*imageReconstruct2[:,j+step_fwd])/(N_interps+1)


    
