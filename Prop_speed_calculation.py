# -*- coding: utf-8 -*-
"""
DATE:           Wed Jun 25 09:36:27 2025
AUTHOR:         Joseph Mockler
DESCRPITION:    This program accepts NN-based 2nd mode WP classification results
                and a set of frames to measure propogation speed in the BL. There
                are three main approaches: 1D correlation around the localized WP,
                2D image correlation analysis around the localized WP, and a 2D 
                windowing approach to develop the correlation vs lag plot. All 
                three are demonstrated in this script and yield consistent results.
                Additionally, this script demonstrates how to first fitler the 
                results in space and time to build a stronger consensus for a WP, 
                which in turn improves the localization estimate.
"""

import os

# Update to project folder location
base_folder = 'C:\\Users\\tyler\\Desktop\\NSSSIP25\\Machine Learning Classification - NSSSIP25'

os.chdir(base_folder)

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from ML_utils import *
from Results_utils import *
from Propagation_speed_functions import *

#%% I. LOAD EXISTING CLASSIFICATION RESULTS
# Load in a classified video
Confidence_history = np.load(f'{base_folder}\\Example Data and Outputs\\Classify_a_video\\classification_results_LangleyRun34_filtered.npy')

print('Reading training data file')

# Confidence history should be a numpy array of np.arrays!
# The inner arrays are the [0, 1] (NOT binary yet!) confidence results from the 
# NN classification, where the length of each inner list corresponds to the sliced
# images in a single frame. 

# This can be done with Classify_a_video.py!

# Load in the classified image data
file_name = f"{base_folder}\\Example Data and Outputs\\video_data_LangleyRun34_105_116ms.txt"
lines, lines_len = write_text_to_code(file_name)

# Lines should correspond to the labeled classifications from Confidence_history
# These MUST match, or else your prop speed results will be junk!

#%% II. FILTER THE WP ID'S IN TIME AND SPACE

# Define the post-processing parameters you want to use
window_size = 3                 # Number of windows to compute moving confidence
confid_thres = 1.5              # Summed threshold over summed window
indiv_thres = 0.9               # Individual exception to windowing threshold
use_post_process = 1
slice_width = 64                # Frame slice width - make the same as your NN training slice width!!!

# How many frames to process for convection speed?
N_Frames_process = lines_len - 5 # Give some small buffer because of look-ahead procedure

# Define the filtering-in-time parameters
mm_pix = 0.0756                 # Camera/lens characteristic
FR = 285e3                      # Camera frame rate in Hz
dt = 1/FR                       # time step between frames
prop_speed = 825                # A priori estimate of propogation speed
pix_tr_per_frame = round(prop_speed * dt * 1/(mm_pix*1e-3))

# Define the post-processing in time parameters
lookahead_correction = round(pix_tr_per_frame/slice_width)
num_lookahead = 3           # number of frames to lookahead for temporal consensus
use_temporal_process = 1    # Use the filtering in time post processing?

# Now run the convection speed analysis!
WP_locs_list = []
for i in range(N_Frames_process):
    # Start by classifying i'th frame
    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i, lines, 64)
    confidence = Confidence_history[i]
    
    # Now we filter in space (via the windowing technique)
    classification_result, n11, n10, n01, n00 = classify_WP_frame(Imagelist, WP_io, confidence, window_size, indiv_thres, confid_thres, use_post_process)
    
    # Define a tuple of spatial post-processing parameters to pass in
    post_process_params = window_size, indiv_thres, confid_thres, use_post_process
    
    # Define the tuple of temporal post-processing parameters to pass in
    time_process_params = lookahead_correction, num_lookahead
    
    # Now filter in time as well! This extracts the start, stop slices
    start_slice, stop_slice = filter_in_time(Imagelist, Confidence_history, classification_result, i, lines, post_process_params, time_process_params)
    
    # Print the status
    if i % 100 == 0:
        print(f"Frame {i} processed. {100*i/N_Frames_process:.2f}% complete.")
    
    # For a list of start-stop slices for the entire analyzed video set
    WP_locs_list.append([start_slice, stop_slice])        
          
    
#%% III. PERFORM PROP SPEED ANALYSIS VIA 1D CORRELATION

# First calculate approx how many pixels a wave will propogate in a single frame
prop_speed_low = 500                # Lower cutoff for assumed prop speed, m/s
prop_speed_high = 1250              # Upper cutoff for assumed prop speed, m/s
row_search = 42                     # Row to take measurements through - you'll have to dial this in!

pix_tr_low = prop_speed_low * dt * 1/(mm_pix*1e-3)  
pix_tr_high = prop_speed_high * dt * 1/(mm_pix*1e-3)

# Processing parameters
use_bandpass_filter = 1             # Use the BP filter - recommended! But set to 0 if already pre-filtered
BL_height = 19                      # Pixel height of boundary layer - used for bandpass filtering
window_polyfit = 5                  # Number of pts to consider around the max for polyfit - this is JUST one-sided around the max
signal_buffer = 2                   # How much extra "slice" should be kept outside of the localized WP?
corr_max_search = 10                # What range should you search for the max - one-sided
start_lag = -round(pix_tr_high)     # Negative is taken for "backwards" correlation correction
stop_lag = -round(pix_tr_low)
space_connectivity = 3              # Minimum number of connected WP slices to count as a WP (i.e, the min WP length to analyze)

# Use BL_height to calculate bandpass bounds
cut_low = 1/BL_height
cut_high = 1/(4*BL_height)

# Bandpass filter the frames
sos = signal.butter(4, [cut_high, cut_low], btype='bandpass', fs=1, output='sos')

convect_speed = []
convect_breakout = []
for ii in range(N_Frames_process):
        
    # Pick which WP frames we should analyze
    if ii == 0:
        convect_breakout.append([])
        continue
    
    # Check that the WP ii and ii+1 are larg enough to get good results from
    if WP_locs_list[ii][1] - WP_locs_list[ii][0] < space_connectivity or WP_locs_list[ii+1][1] - WP_locs_list[ii+1][0] < space_connectivity:
        convect_breakout.append([])
        continue
    
    # Extract out the starting and stopping values
    WP_loc_ii = WP_locs_list[ii]
    starting_WP = WP_loc_ii[0]
    stopping_WP = WP_loc_ii[1]
    
    start_analysis = max(starting_WP - signal_buffer, 0)
    stop_analysis = min(stopping_WP + signal_buffer, 18)
    
    # Take two frames where we know WP's exist
    Imagelist1, WP_io, slice_width, height, sm_bounds = image_splitting(ii, lines, slice_width)
    Imagelist2, WP_io, slice_width, height, sm_bounds = image_splitting(ii+1, lines, slice_width)
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
print(f'Mean (1D): {mean_speed} +/- {std_dev}')

# Form a histogram and inspect the results
counts, bins = np.histogram([prop*mm_pix *1e-3 / dt for prop in convect_speed], bins=20)
fig, ax1 = plt.subplots(1)
ax1.stairs(counts, bins)
ax1.set_xlabel('Measured propagation speed, m/s')
ax1.set_ylabel('Frequency counts')
ax1.set_xlim(750, 900)
plt.show()

#%% IV. PLOT EXAMPLE RESULTS FOR VISUALIZATION
# Plot two frames where you know WP's exist, and check that the locations appear correct
plot_analyzed_frames(lines, 578, 579, WP_locs_list, row_search, slice_width)

# This forms a slightly more rigorous plot of measured speed vs WP length or confidence
# Useful for refining results to get better prop speed estimates. 
convection_confidence_analysis(convect_breakout, Confidence_history, WP_locs_list)

#%% V. PROP SPEED MEASUREMENT VIA 2D ANALYSES

prop_speed_low = 500                # Lower cutoff for assumed prop speed, m/s
prop_speed_high = 1250              # Upper cutoff for assumed prop speed, m/s

bl_range_start = 40                 # What rows of the frames you should search
bl_range_end = 50

pix_tr_low = prop_speed_low * dt * 1/(mm_pix*1e-3)  # Computed number of pixels traveled between frames
pix_tr_high = prop_speed_high * dt * 1/(mm_pix*1e-3)

# Processing parameters
use_bandpass_filter = 1             # Use the BP filter? Recommended if not already filtered
use_windowing_method = 1            # Uses the windowing approach that Dr. Laurence recommended
window_size = 64                    # Windowing size for Dr. Laurence's approach

use_2D_corr_method = 0              # Uses the 2D image correlation approach
window_polyfit = 5                  # Window for computing convection speed from polyfit
signal_buffer = 0                   # How much extra "slice" should be kept outside of the localized WP?
corr_max_search = 10                # What range should you search for the max


start_lag = -round(pix_tr_high)     # Negative is taken for "backwards" correlation correction
stop_lag = -round(pix_tr_low)

# Bandpass filter the frames
sos = signal.butter(4, [cut_high, cut_low], btype='bandpass', fs=1, output='sos')

convect_speed = []
convect_breakout = []
for k in range(N_Frames_process):
    
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
    
    start_analysis = max(starting_WP - signal_buffer, 0)
    stop_analysis = min(stopping_WP + signal_buffer, 18)
    
    # Take two frames where we know WP's exist
    Imagelist1, WP_io, slice_width, height, sm_bounds = image_splitting(k, lines, slice_width)
    Imagelist2, WP_io, slice_width, height, sm_bounds = image_splitting(k+1, lines, slice_width)
    imageReconstruct1 = np.hstack(Imagelist1[start_analysis:stop_analysis])     # WP's should be at 10 to 18
    imageReconstruct2 = np.hstack(Imagelist2[start_analysis:stop_analysis])     # WP's should be at 10 to 18
    
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
        subpix_convect = -pfit[1]/(2*pfit[0])
        
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

print(f'Mean (2D): {mean_speed} +/- {std_dev}')

# Form a histogram and inspect the results - x-bounds based on prop_speed_low and prop_speed_high
counts, bins = np.histogram([prop*mm_pix *1e-3 / dt for prop in convect_speed], 20)
fig, ax1 = plt.subplots(1)
ax1.stairs(counts, bins)
ax1.set_xlabel('Measured propagation speed, m/s')
ax1.set_ylabel('Frequency counts')
ax1.set_xlim(700, 1000)
plt.show()

#%% Save to .mat file for nice plotting
from scipy.io import savemat
convect_speed_physical = [prop*mm_pix *1e-3 / dt for prop in convect_speed]

savemat(f"{base_folder}\\Example Data and Outputs\\Prop_speed_calculation\\TESTRun34_histogram_data.mat", {'Run34_convect_speeds': convect_speed_physical})

#%% VI. PERFORM PROP SPEED ANALYSIS VIA OPTICAL FLOW
# First calculate approx how many pixels a WP will propogate in a single frame
prop_speed_low = 650                # Lower cutoff for assumed prop speed, m/s
prop_speed_high = 1000              # Upper cutoff for assumed prop speed, m/s

pix_tr_low = round(prop_speed_low * dt * 1/(mm_pix*1e-3))
pix_tr_high = round(prop_speed_high * dt * 1/(mm_pix*1e-3))

# Initialize vectors to save measurements
prop_speed_mps, prop_speed_pix = [], []

# Initialize subroutine lists
measured_prop = []
time_vec = []
sos = signal.butter(4, [0.0133, 0.10], btype='bandpass', fs=1, output='sos')
print_flag = 0

st_time = time.time()
N_Frames_process = lines_len - 6 
for k in range(N_Frames_process):
    
    # Print the status
    if k % 100 == 0:
        print(f"Frame {k} processed. {100*k/N_Frames_process}% complete.")
    
    displacements = []
    i1, i2 = k, k+1
    skip_flag = 0
    
    # Run the offset-based optical flow calculation to estimate prop speed
    for offset in range(pix_tr_low,pix_tr_high):
        
        fr1_WP = WP_locs_list[i1]
        fr2_WP = WP_locs_list[i2]
        
        # First check if the two consectutative frames actually have WP's
        if (fr1_WP[1] - fr1_WP[0] == 0) or (fr2_WP[1] - fr2_WP[0] == 0):
            skip_flag = 1
            continue
        
        # Provide a conservative estimate of the WP bounds because our optical flow 
        # results should only be looking in the areas of definite WP
        start_col, stop_col = max(fr1_WP[0], fr2_WP[0])*slice_width, min(fr1_WP[1], fr2_WP[1])*slice_width
        
        # Skip if there's no data to look at after the WP searching
        if stop_col - start_col < 1:
            skip_flag = 1
            continue
        
        # Only consider a portion of the BL
        start_row, stop_row = 30, 55
        
        # Now read off the frames
        Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i1, lines, slice_width)
        frame1 = np.hstack(Imagelist)
    
        Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i2, lines, slice_width)
        frame2 = np.hstack(Imagelist)
        
        # Filter in space
        row_range = frame1.shape[0]
        col_range = frame1.shape[1]
        
        st_time = time.time()
        for i in range(row_range):
            frame1[i,:] = signal.sosfiltfilt(sos, frame1[i,:])
            frame2[i,:] = signal.sosfiltfilt(sos, frame2[i,:])
    
        # Normalize the frame to adjust contrast and convert to 8bit (required by opt. flow implementation)
        frame1 = np.uint8(cv.normalize(frame1, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX))
        frame2 = np.uint8(cv.normalize(frame2, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX))
        
        # Now only take the regions of the image we care about
        # Adjust the second frame backwards
        frame1 = frame1[start_row:stop_row, start_col:stop_col]
        frame2 = frame2[start_row:stop_row, start_col+offset:stop_col+offset]
    
        # now try bringing frame 2 back by the offset
        frame2[:,start_col:stop_col] = frame2[:,start_col:stop_col+offset]
        frame2 = np.delete(frame2, slice(stop_col, stop_col+offset), axis = 1)
        
        # Compute the optical flow
        flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 2, 5, 3, 7, 1.1, 0)
        u = flow[:,:,0]
        v = flow[:,:,1]
        end_time = time.time()
        time_vec.append(end_time-st_time)
        
        # Compute the mean flow across the displacements
        displacements.append(np.median(u))
        #measured_prop.append(displacements[offset] + offset)
    
    # Now compute the linear fit to solve for subpixel displacement
    if skip_flag == 0:
        lin_fit = np.polyfit(range(pix_tr_low,pix_tr_high), displacements, 1)
        est_prop = -lin_fit[1]/lin_fit[0]
        
        mm_pix = 0.0756         # From paper
        FR = 285e3              # Camera frame rate in Hz
        dt = 1/FR               # time step between frames
        
        if print_flag == 1:
            print(f'Est prop speed: {est_prop} pixels')
            print(f'Est prop speed: {est_prop*mm_pix*1e-3/dt} m/s')
            
        prop_speed_pix.append(est_prop)
        prop_speed_mps.append(est_prop*mm_pix*1e-3/dt)

# Filter spurious results
prop_speed_mps_filt = []
for meas in prop_speed_mps:
    if meas < -prop_speed_high or meas > -prop_speed_low:
        continue
    prop_speed_mps_filt.append(meas)
    
counts, bins = np.histogram(prop_speed_mps_filt, 20)
fig, ax1 = plt.subplots(1)
ax1.stairs(counts, bins)
ax1.set_xlabel('Measured propagation speed, m/s')
ax1.set_ylabel('Frequency counts')
#ax1.set_xlim(700, 1000)
#ax1.set_xlim(-1000, -700)
plt.show()

print(f'Mean: {np.mean(prop_speed_mps_filt)} +/- {np.std(prop_speed_mps_filt)}')

# This forms a slightly more rigorous plot of measured speed vs WP length or confidence
# Useful for refining results to get better prop speed estimates. 
convection_confidence_analysis(prop_speed_mps, Confidence_history, WP_locs_list)


#%% Save off some figures for poster
from matplotlib.patches import Rectangle
start_vid, stop_vid =  980, 1008
for i_iter in range(start_vid, stop_vid, 4):
    # Split the image and classify the slices
    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines, slice_width)

    imageReconstruct = np.hstack([image for image in Imagelist])
    
    fig, ax = plt.subplots(1, figsize=(16, 4))
    ax.imshow(imageReconstruct, cmap = 'gray')
    
    # classification_result = filtered_result_history[i_iter]
    
    WP_locs_plot = WP_locs_list[i_iter]
    WP_start = WP_locs_plot[0]
    WP_end = WP_locs_plot[1]
    WP_width = WP_end-WP_start
    
    rect = Rectangle((WP_start*slice_width, 1), WP_width*slice_width, height-3,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    ax.set_title('Frame '+str(i_iter))
    
    # Add the ground truth
    #ax.add_patch(Rectangle((sm_bounds[0],sm_bounds[1]), sm_bounds[2], sm_bounds[3], edgecolor='blue', facecolor='none'))
    ax.tick_params(axis='both', labelsize=8) # Change 8 to your desired smaller size
    # plt.savefig("C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\Poster plotting\\CF33\\plotted_img"+str(i_iter)+".svg")
    # savemat("C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\Poster plotting\\Run33_frame"+str(i_iter)+".mat", {"Run34_frame"+str(i_iter): imageReconstruct})
    savemat(f"{base_folder}\\Example Data and Outputs\\Prop_speed_calculation\\Poster plotting\\TESTRun34_frame"+str(i_iter)+"WPs.mat", {"Run34_frame"+str(i_iter)+"WPs": WP_locs_plot})
    plt.show()
    


