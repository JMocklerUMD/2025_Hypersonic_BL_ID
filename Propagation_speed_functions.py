# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 08:28:35 2025

@author: Joseph Mockler
"""

from Results_utils import *
from ML_utils import *
import numpy as np
from scipy import signal
import math


def filter_in_time(Imagelist, Confidence_history, filtered_result, i, lines, post_process_params, time_process_params):
    
    # Pull out required parameters
    window_size, indiv_thres, confid_thres, use_post_process = post_process_params
    lookahead_correction, num_lookahead = time_process_params
    
    # Perform a "lookahead" of the next 3 frames
    # Sum the classifications along the time dimension 
    # (e.g form a 19x1 array that's the sum 4 time steps of classification results)
    WP_locs = np.zeros(len(Imagelist))
    WP_locs = WP_locs + filtered_result
    for ii in range(1,num_lookahead+1):
        Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i + ii, lines, len(Imagelist[0]))
        confidence = Confidence_history[i+ii]
        
        classification_result, n11, n10, n01, n00 = classify_WP_frame(Imagelist, WP_io, confidence, window_size, indiv_thres, confid_thres, use_post_process)
        
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
            if WP_locs[start_slice-lookahead_correction] > 0:
                start_slice = start_slice - 1
                
            # Handle when the WP is at the start of the frame
            if start_slice < 0:
                start_slice = 0
            break # break the enumerate(WP_locs) for loop
    
        # Handle advection off the screen
        if consec == 1 and j == len(WP_locs) - 1:
            stop_slice = j
            
    return start_slice, stop_slice

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
    
def plot_analyzed_frames(lines, i, ii, WP_locs_list, row_search, slice_width):

    start_analysis = WP_locs_list[i][0]
    stop_analysis = WP_locs_list[ii][1]
    Imagelist1, WP_io, slice_width, height, sm_bounds = image_splitting(i, lines, slice_width)
    Imagelist2, WP_io, slice_width, height, sm_bounds = image_splitting(ii, lines, slice_width)
    imageReconstruct1 = np.hstack(Imagelist1[start_analysis:stop_analysis])     # WP's should be at 10 to 18
    imageReconstruct2 = np.hstack(Imagelist2[start_analysis:stop_analysis])     # WP's should be at 10 to 18
    
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
    plt.show()
    
def convection_confidence_analysis(convect_breakout, Confidence_history, WP_locs_list):
    # Does WP size correlate with measurement performance?
    fig, ax1 = plt.subplots(1, figsize = (13,5))
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
        for j in range(locs[0], locs[1]):
            confid_list.append(confid[j])
            
        mean_confid.append(np.mean(confid_list))
        mean_convect_frame.append(mean_meas)
        std_convect_frame.append(std_dev_meas)
    
    fig, ax1 = plt.subplots(1, figsize = (13,5))
    ax1.scatter(mean_confid, mean_convect_frame)
    ax1.set_xlabel('WP Classifier Confidence')
    ax1.set_ylabel('Mean prop speed (pix) between 2 frames')
    #ax1.set_ylim(-70, -10)
    plt.show()