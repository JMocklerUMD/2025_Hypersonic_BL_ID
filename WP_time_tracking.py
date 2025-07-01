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


#%% Load classification results + video
Confidence_history = np.load('C:\\Users\\Joseph Mockler\\Run34_video.npy')

print('Reading training data file')

# Write File Name
file_name = 'C:\\UMD GRADUATE\\RESEARCH\\Hypersonic Image ID\\videos\\Test1\\run34\\full_video_data.txt'
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

for i in range(1995):
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
        if WP_candidate < 2 and consec == 1:
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
    # print(f"Frame {i}: WP_loc = [{start_slice}, {stop_slice}]")          
    
    
#%% Perform 1D propagation analysis via correlation - necessary fcns

def subpixel_convection_speed(lags, corr, window_polyfit):
    # Find the max points
    corr_idx = np.argmax(corr) # Note that corrmax and lagmax share the same idx
    # lag_max = lags[np.argmax(corr)]
    
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
prop_speed = 800        # A priori estimate of propogation speed
pix_tr = prop_speed * dt * 1/(mm_pix*1e-3)  # Computed number of pixels traveled between frames

# Calculate how much of the buffer we're throwing out and set the number of increments
cutoff_len = round(pix_tr)
row_search = 42
N_interps = 3
increment = round(pix_tr/(N_interps+1))

# Processing parameters
window_polyfit = 5      # Window for computing convection speed from polyfit
signal_buffer = 2       # How much extra "slice" should be kept outside of the localized WP?

# Bandpass filter the frames
sos = signal.butter(4, [0.0196, 0.1176], btype='bandpass', fs=1, output='sos')

convect_speed = []
convect_breakout = []
for ii in range(1995):
    
    # Pick which WP frames we should analyze
    if ii == 0:
        convect_breakout.append([])
        continue
    
    # WP is too small to consider
    if WP_locs_list[ii][1] - WP_locs_list[ii][0] < 4 or WP_locs_list[ii-1][1] - WP_locs_list[ii-1][0] < 4:
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
    line1 = signal.sosfiltfilt(sos, line1)
    line2 = signal.sosfiltfilt(sos, line2)
    
    # Now form the signal interpolation based on the a priori propogation speed
    # Need a (N-1) x length array to store the intermediate 
    recreated_signals = np.zeros((N_interps, len(line1[:-cutoff_len])))
    
    for i in range(1,N_interps+1):
        for j in range(len(line1)-cutoff_len):
            # Look backwards i*increment from the
            step_bwd = i*increment
            step_fwd = ((N_interps+1)-i)*increment
            recreated_signals[i-1,j] = (((N_interps+1)-i)*line1[j-step_bwd] + i*line2[j+step_fwd])/(N_interps+1)
    
    # Do the first interpolation
    line1 = line1[:-cutoff_len]
    line2 = line2[:-cutoff_len]
    
    # Compute the cross-correlation
    lags, corr = correlate_1D_signals(line1, recreated_signals[0,:])
    
    # Calculate the convection speed via a correlation analysis
    subpix_convect = subpixel_convection_speed(lags, corr, window_polyfit)
    convect_speed.append(subpix_convect)
    
    # Do the middle ones
    for i in range(N_interps-1):
        # Compute the cross-correlation
        lags, corr = correlate_1D_signals(recreated_signals[i,:], recreated_signals[i+1,:])
        
        # Calculate the convection speed via a correlation analysis
        subpix_convect = subpixel_convection_speed(lags, corr, window_polyfit)
        convect_speed.append(subpix_convect)
        
    # Do the final
    # Compute the cross-correlation
    lags, corr = correlate_1D_signals(recreated_signals[N_interps-1,:], line2)
   
    # Calculate the convection speed via a correlation analysis
    subpix_convect = subpixel_convection_speed(lags, corr, window_polyfit)
    convect_speed.append(subpix_convect)
    
    # Append to a master list for later analysis
    convect_breakout.append(convect_speed[-(N_interps+1):])

# Perform some basic statistics
mean_speed = np.mean(convect_speed)*mm_pix *1e-3 / (dt/(N_interps+1))
std_dev = np.std(convect_speed)*mm_pix *1e-3 / (dt/(N_interps+1))

print(f'Cut numbers: {N_interps}, Row number: {row_search}, Approx prop speed: {prop_speed}')
print(f'Mean: {mean_speed} +/- {std_dev}')
#print(convect_breakout)

# Form a histogram and inspect the results
counts, bins = np.histogram(convect_speed)
fig, ax1 = plt.subplots(1)
ax1.stairs(counts, bins)
ax1.set_xlabel('Measured propagation speed, pixels')
ax1.set_ylabel('Frequency counts')
plt.show()

#%% Data save
# convect_900 = convect_speed

# #%% Bar chart of classification results
# counts750, bins750 = np.histogram(convect_750, bins = 2)
# counts800, bins800 = np.histogram(convect_800, bins = 2)
# counts850, bins850 = np.histogram(convect_850, bins = 2)
# counts900, bins900 = np.histogram(convect_900, bins = 2)

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (14,4))
# ax1.stairs(counts750, bins750)
# ax1.set_xlim(-8, -5)
# ax1.set_ylabel("Counts of measured prop speed")
# ax2.stairs(counts800, bins800)
# ax2.set_xlim(-8, -5)
# ax3.stairs(counts850, bins850)
# ax3.set_xlim(-8, -5)
# ax4.stairs(counts900, bins900)
# ax4.set_xlim(-8, -5)
#plt.stairs(counts, bins)

#%% correlation plot thing
# fig, ax = plt.subplots(1)
# ax.plot(lags, corr)
# ax.set_ylabel('Computed Correlation')
# ax.set_xlabel('Pixel lead/lag')
# ax.axvline(x=lags[np.argmax(corr)], color='r', linestyle='-')
# ax.set_xlim(-100,100)
# plt.show()


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
ax2.scatter(mean_confid, std_convect_frame)
ax2.set_xlabel('WP Classifier Confidence')
ax2.set_ylabel('Stdev prop speed (pix) between 2 frames')
plt.show()

#%% 2D correlation method
# First calculate approx how many pixels a wave will propogate in a single frame
mm_pix = 0.0756         # From paper
FR = 285e3              # Camera frame rate in Hz
dt = 1/FR               # time step between frames
prop_speed = 900        # A priori estimate of propogation speed
pix_tr = prop_speed * dt * 1/(mm_pix*1e-3)  # Computed number of pixels traveled between frames

# Calculate how much of the buffer we're throwing out and set the number of increments
cutoff_len = round(pix_tr)
row_search = 42
N_interps = 3
increment = round(pix_tr/(N_interps+1))
convect_speed = []
convect_breakout = []

for k in range(67, 68):
    
    # This stuff just does filtering if we should bother analyzing this frame
    if k == 0:
        convect_breakout.append([])
        continue
    
    # WP is too small to consider
    if WP_locs_list[k][1] - WP_locs_list[k][0] < 4 or WP_locs_list[k-1][1] - WP_locs_list[k-1][0] < 4:
        convect_breakout.append([])
        continue
    
    WP_loc_ii = WP_locs_list[ii]
    starting_WP = WP_loc_ii[0]
    stopping_WP = WP_loc_ii[1]
    
    start_analysis = max(starting_WP - 2, 0)
    stop_analysis = min(stopping_WP + 2, 18)
    
    # Take two frames where we know WP's exist
    Imagelist1, WP_io, slice_width, height, sm_bounds = image_splitting(k, lines)
    Imagelist2, WP_io, slice_width, height, sm_bounds = image_splitting(k+1, lines)
    imageReconstruct1 = np.hstack(Imagelist1[start_analysis:stop_analysis])     
    imageReconstruct2 = np.hstack(Imagelist2[start_analysis:stop_analysis])     
    
    row_range, col_range = imageReconstruct1.shape
    
    # Bandpass filter the entire frame
    for i in range(row_range):
        imageReconstruct1[i,:] = signal.sosfiltfilt(sos, imageReconstruct1[i,:])
        imageReconstruct2[i,:] = signal.sosfiltfilt(sos, imageReconstruct2[i,:])
    
    # Our recreated signals should chop off the front and back bit to avoid
    # any weird clipping I noticed
    recreated_signals = np.zeros((N_interps, row_range, col_range))
    
    # ii forms the intermediate frames
    for ii in range(1,N_interps+1):
        
        # j iterates over the cols (weighted avg is equally applied across the entire col)
        for j in range(cutoff_len, col_range-cutoff_len):
            # Look backwards i*increment from the
            step_bwd = ii*increment
            step_fwd = ((N_interps+1)-ii)*increment
            recreated_signals[ii-1,:,j] = (((N_interps+1)-ii)*imageReconstruct1[:,j-step_bwd] + ii*imageReconstruct2[:,j+step_fwd])/(N_interps+1)
    
    # Reshape the frames for the cutoff locs we described above
    img1 = imageReconstruct1[:, cutoff_len:-cutoff_len]
    img2 = imageReconstruct2[:, cutoff_len:-cutoff_len]
    recreated_signals = recreated_signals[:,:,cutoff_len:-cutoff_len]
    
    # Find the first correlation
    lags2 = signal.correlation_lags(len(img1[1,:]), len(recreated_signals[0,1, :]), mode = "same")
    corr2 = signal.correlate2d(img1, recreated_signals[0,:, :], boundary='symm', mode='same')
    # corr2_check = corr2
    row_max, col_max = np.unravel_index(np.argmax(corr2), corr2.shape)
    convect_speed.append(lags2[col_max])
    
    # Do the middle ones
    for i in range(N_interps-1):
        lags2 = signal.correlation_lags(len(recreated_signals[i,1, :]), len(recreated_signals[i+1,1, :]), mode = "same")
        corr2 = signal.correlate2d(recreated_signals[i,:, :], recreated_signals[i+1,:, :], boundary='symm', mode='same')
        #print(f"t0 to t1 pixel lag {lags[np.argmax(corr)]}. Speed = {lags[np.argmax(corr)]*mm_pix *1e-3 / (dt/(N_interps+1))}")
        if i == 0:
            corr2_check= corr2
        row_max, col_max = np.unravel_index(np.argmax(corr2), corr2.shape)
        convect_speed.append(lags2[col_max])
        
    # Do the final
    lags2 = signal.correlation_lags(len(recreated_signals[N_interps-1,1,:]), len(img2[1,:]), mode = "same")
    corr2 = signal.correlate2d(recreated_signals[N_interps-1,:, :], img2, boundary='symm', mode='same')
    row_max, col_max = np.unravel_index(np.argmax(corr2), corr2.shape)
    convect_speed.append(lags2[col_max])
    
        
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize = (15,10))
    ax1.imshow(img1, cmap = 'gray')
    ax1.set_title('Frame 1')
    ax2.imshow(recreated_signals[0,:,:], cmap = 'gray')
    ax2.set_title('Frame 1 + 2us')
    ax3.imshow(img2, cmap = 'gray')
    ax3.set_title('Frame 2')
    ax4.imshow(corr2, cmap = 'gray')
    ax4.set_title('Frame 1 + 2us and Frame 1 correlation')
    plt.show()
    
    print(f"Frame: {k}. Measured convection speeds {convect_speed[-(N_interps+1):]}")
    convect_breakout.append(convect_speed[-(N_interps+1):])

mean_speed = np.mean(convect_speed)*mm_pix *1e-3 / (dt/(N_interps+1))
std_dev = np.std(convect_speed)*mm_pix *1e-3 / (dt/(N_interps+1))

print(f'Cut numbers: {N_interps}, Approx prop speed: {prop_speed}')
print(f'Mean: {mean_speed} +/- {std_dev}')

# 



    
