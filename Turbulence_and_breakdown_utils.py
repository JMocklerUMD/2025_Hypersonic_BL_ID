# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 10:11:00 2025

@author: Tyler Ostrowski
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

def initialize_Langley_vals():
    '''
    Initializes mm_pix, FR, start_mm for Langley runs from https://link.springer.com/article/10.1007/s00348-023-03758-w
    Initializes approximate prop_speed for Langley Re9.7 from above paper

    OUTPUTS
    -------
    mm_pix : float
        mm to pixel conversion for camera set up in above paper
    FR : float
        Camera frame rate in Hz
    prop_speed : int
        A priori estimate of propagation speed, m/s, for Re9.7
    start_mm : int or float
        mm location on cone where sclieren field of view begins

    '''
    mm_pix = 0.0756     
    FR = 285e3          
    prop_speed = 825      
    start_mm = 288
    return mm_pix, FR, prop_speed, start_mm

def intermittency_plot(frame_res_history,start_mm,mm_pix,slice_width):
    """
    Plots an intermittency plot for the detected feature.
    Assumes all frames are the same length.

    INPUTS
    ----------
    frame_res_history : array of arrays for every frame; arrays for each frame have length equal to the number of slices (num_slices)
        list of numpy arrays with post-processed slice classifications for each frame
    
    start_mm: int or float,
        downstream mm location on the object surface where Schlieren image begins
        
    mm_pix: int or float,
        mm/pixel conversation factor for camera/lens set up
        
    slice_width: int,
        width of image slices in pixels

    OUTPUTS:
    -------
    None.
    Plots result.
    
    Reference:
    "Global analysis of nonlinear second-mode development in a Mach-6 boundary layer from high-speed schlieren data"
    pg. 24
    """
    num_slices = len(frame_res_history[0])
    N_frames = len(frame_res_history)

    intermittency = np.zeros(num_slices)
    
    for i_iter in range(N_frames):
        for i, cls_result in enumerate(frame_res_history[i_iter]):
            if cls_result == 1:
                intermittency[i] = intermittency[i] + 1
                
    intermittency = intermittency / N_frames
    
    extent = num_slices*slice_width*mm_pix
    max_extent = start_mm+extent
    mm_width = slice_width*mm_pix
    
    x_values = [] # use the middle of slice
    for x in np.arange(start_mm+mm_width/2.0,max_extent,mm_width):
        x_values.append(x)
    x_values = np.array(x_values)
        
    plt.scatter(x_values, intermittency)
    plt.xlabel('Downstream Location (mm)')
    plt.ylabel('Intermittency')
    plt.title('Intermittency over Object Length')
    
    plt.show()
    return x_values, intermittency
    

def calc_prop_speed_pix(mm_pix,FR,prop_speed):
    """
    Calculate the propogation speed in pixels/frame

    INPUTS
    ----------
    mm_pix: int or float,
        mm/pixel conversation factor for camera/lens set up
        
    FR : float
        Camera frame rate in Hz
        
    prop_speed : int or float
        A priori estimate of propagation speed, m/s

    OUTPUTS
    -------
    PS_pix_frame : float
        propagation speed in pixels/frame

    """
    dt = 1/FR                     # Time step between frames
    PS_pix_frame = prop_speed * dt * 1/(mm_pix*1e-3)  # Computed number of pixels traveled between frames
    return PS_pix_frame

    
def breakdown_stats(wp_result_history,turb_result_history,slice_width,PS_pix_frame):
    '''

    Finds the percentage of turbulence that came from (1) second-mode wave packets, (2) upstream, or (3) an uncertain source. 
    Finds sequences of frames where wave packets broke down into turbulence.
    Finds sequences of frames where turbulence came into frame from upstream.

    INPUTS
    ----------
    wp_result_history : array of arrays for every frame; arrays for each frame have length equal to the number of slices (num_slices)
        list of numpy arrays with post-processed wave packet slice classifications for each frame
    turb_result_history : array of arrays for every frame; arrays for each frame have length equal to the number of slices (num_slices)
        list of numpy arrays with post-processed turbulence slice classifications for each frame
    slice_width: int,
        width of image slices in pixels.
    PS_pix_frame : float
        propagation speed in pixels/frame

    OUTPUTS
    -------
    breakdown_sequences : (A,2) array
        array with pairs of numbers representing the start and end frame index number for a sequence showing a wave packet breaking down into turbulence
    upstream_turb_sequences : (B,2) array
        array with pairs of numbers representing the start and end frame index number for a sequence showing turbulence flow in from upstream

    '''
    overwritten_turb_hist = copy.deepcopy(turb_result_history)
    N_frames = len(turb_result_history)
    
    breakdown_sequences = []
    upstream_turb_sequences = []
    overwrite_record = []
    total_turb_count = 0
    from_wp = 0
    from_upstream_turb = 0
    from_inconclusive = 0
    
    for i_iter in range(N_frames-1,0,-1): #go through frames backwards
        min_iter = N_frames
        breakdown_observed = 0
        upstream_turb = False
        
        for i, turb_result in enumerate(overwritten_turb_hist[i_iter]):
            if turb_result == 1:
                total_turb_count = total_turb_count + 1
                # track turbulence
                k_iter, k, overwite_array = track_slice(N_frames,i_iter,i,overwritten_turb_hist,slice_width,PS_pix_frame,overwrite = True)
                overwrite_record.append(overwite_array)
                # then track wave packets
                g_iter, g, overwite_array = track_slice(N_frames,k_iter,k,wp_result_history,slice_width,PS_pix_frame)
                
                '''
                # Troubleshooting outputs
                print(f'i: {i}')
                print(f'k: {k}')
                print(f'g: {g}')
                print(f'i_iter: {i_iter}')
                print(f'k_iter: {k_iter}')
                print(f'g_iter: {g_iter}')
                print(f'len(overwritten_turb_hist[g_iter]): {len(overwritten_turb_hist[g_iter])}')
                '''
                
                if k == 0:
                    # went all the way to beginning of frame and was alway turbulent
                    from_upstream_turb = from_upstream_turb + 1
                    upstream_turb = True
                elif g_iter == 0: 
                    # went all the way to the first frame
                    from_inconclusive = from_inconclusive + 1
                elif k==g:
                    # no detected WPs after detecting turbulence
                    from_inconclusive = from_inconclusive + 1
                else:
                    from_wp = from_wp + 1
                    breakdown_observed = breakdown_observed + 1
                
                # base frame sequence over turblent slice tracked the longest
                if g_iter < min_iter:
                    min_iter = g_iter

        if breakdown_observed >= 3: # must observe a certain number of occurance to store sequence
            breakdown_sequences.append([i_iter, min_iter]) 
        if upstream_turb:
            upstream_turb_sequences.append([i_iter, min_iter])             
        overwrite(overwrite_record, overwritten_turb_hist)
        
    print('------Observed Origin of Turbulence------') 
    print(f'Second mode wave packets: {from_wp/total_turb_count*100}%')
    print(f'Upstream turbulence: {from_upstream_turb/total_turb_count*100}%')
    print(f'Inconclusive: {from_inconclusive/total_turb_count*100}%')
    
    return breakdown_sequences, upstream_turb_sequences
                

def track_slice(N_frames,i_iter,i,result_history,slice_width,PS_pix_frame,overwrite = False):
    '''

    INPUTS
    ----------
    N_frames : int
        total number of frames in dataset
    i_iter : int
        starting frame index number
    i : int
        starting slice index number
    result_history : array of arrays for every frame; arrays for each frame have length equal to the number of slices (num_slices)
        list of numpy arrays with post-processed slice classifications for each frame
    slice_width: int,
        width of image slices in pixels
    PS_pix_frame : float
        propagation speed in pixels/frame
    overwrite : conditional, optional
        whether or not to overwrite values. Used only for turbulence to avoid repeated detections. The default is False.

    OUTPUTS
    -------
    k_iter : int
        final frame index number after tracking the sequence
    k : int
        final slice index number after tracking the sequence
    overwrite_array : (C,2) array
        array with index frame number, index slice number pairs to record which slices should be overwritten

    '''
    max_patience = 5            # how long of a gap between subsequent detections is allowed
    patience = max_patience
    shift = 0
    overwrite_array = []
    
    #prevent errors if the for loop never runs
    j_iter = i_iter
    
    for j_iter in range(i_iter-1,-1,-1): # search back in time through frames
        if patience > 0:
            if i-shift >= 0 and i-shift < len(result_history[0]):
                shift = shift + int(slice_width//PS_pix_frame)    # shift over which slice is checked based on WP propagation speed
                if result_history[i_iter][i-shift] == 1:          # if feature detected...
                    patience = max_patience                       # reset patience
                    if overwrite:                                 # store overwrite location if applicable
                        i_iter = j_iter
                        i = i - shift
                        overwrite_array.append([i_iter,i])
                else:
                    patience = patience - 1                      # otherwise reduce patience
            else:
                
                break
        else:
            break
                    
    if patience > 3: # adjust for patience to return last slice with detected feature      
        diff_patience = max_patience - patience           
        k_iter = j_iter - diff_patience
        k = i - shift - diff_patience
    else:
        k_iter = j_iter
        k = i - shift
    
    return k_iter, k, overwrite_array


def overwrite(overwrite_record, overwritten_turb_hist):
    for _,set in enumerate(overwrite_record):
        for _,[w_iter,w] in enumerate(set):
            overwritten_turb_hist[w_iter][w] = 0
    
    
