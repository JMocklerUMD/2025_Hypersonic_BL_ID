# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 09:36:27 2025

@author: Joseph Mockler
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

#%%

window_size = 3
confid_thres = 1.5
indiv_thres = 0.9
use_post_process = 1

for i in range(100):
    # Start by classifying i'th frame
    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i, lines)
    confidence = Confidence_history[i]
    classification_result, filtered_result, n00, n01, n10, n11 = classify_the_frame(Imagelist, confidence, WP_io, window_size, indiv_thres, confid_thres, use_post_process)
    

    # Perform a "lookahead" of the next 3 frames
    WP_locs = np.zeros(len(Imagelist))
    WP_locs = WP_locs + filtered_result
    for ii in range(1,4):
        Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i + ii, lines)
        confidence = Confidence_history[i+ii]
        classification_result, filtered_result, n00, n01, n10, n11 = classify_the_frame(Imagelist, confidence, WP_io, window_size, indiv_thres, confid_thres, use_post_process)
        
        # Add to the next frame
        WP_locs = WP_locs + filtered_result
        #print(filtered_result)
        
    # Now go thru the list and ID where, among the 4 frames, there's consistency
    start_slice = 0
    stop_slice = 0
    consec = 0
    for j, WP_candidate in enumerate(WP_locs):
        # First see when we detect a WP
        if WP_candidate > 3 and consec == 0:
            start_slice = j
            consec = 1
        
        # Now see when it stops in the frame and be done
        if WP_candidate < 2 and consec == 1:
            consec = 0
            stop_slice = j
            
            # Perform a correction to account for convection
            if WP_locs[start_slice-1] > 0:
                start_slice = start_slice - 1
                
            # Handle when the WP is at the start of the frame
            if start_slice < 0:
                start_slice = 0
                
            break
    
        # Handle advection off the screen
        if consec == 1 and j == len(WP_locs) - 1:
            stop_slice = j
                
            
            
    print(f"Frame {i}: WP_loc = [{start_slice}, {stop_slice}]") 
    #print(WP_locs)           
    
    
#%% Try the correlation thing
from scipy import signal

# First calculate approx how many pixels a wave will propogate in a single frame
mm_pix = 0.0756         # From paper
FR = 258e3              # Camera frame rate in Hz
dt = 1/FR               # time step between frames
prop_speed = 825        # A priori estimate of propogation speed
pix_tr = prop_speed * dt * 1/(mm_pix*1e-3)  # Computed number of pixels traveled between frames

# Calculate how much of the buffer we're throwing out and set the number of increments
cutoff_len = round(pix_tr)
row_search = 40
N_interps = 4
increment = round(pix_tr/N_interps)

# Take two frames where we know WP's exist
Imagelist1, WP_io, slice_width, height, sm_bounds = image_splitting(65, lines)
Imagelist2, WP_io, slice_width, height, sm_bounds = image_splitting(66, lines)
imageReconstruct1 = np.hstack(Imagelist1[6:18])     # WP's should be at 10 to 18
imageReconstruct2 = np.hstack(Imagelist2[6:18])     # WP's should be at 10 to 18

# Take some slices at the boundary layer to perform the correlation
line1 = imageReconstruct1[row_search,:]
line2 = imageReconstruct2[row_search,:]
line_interp = np.zeros(len(line1)-round(pix_tr))

# Now form the signal interpolation based on the a priori propogation speed
N_interps = 4
increment = round(pix_tr/N_interps)

# For each signal along the flow at row 45...
line_t0 = line1[:(len(line1)-round(pix_tr))]

# Iterate thru each desired upsampled curve
# for i in range(1,N_interps):
#     fig, ax = plt.subplots(1, figsize = (15,6))
    
#     # Now use a weighted sum to compute the new interpolated signal
#     for j in range(len(line1)-round(pix_tr)):
#         # Look backwards i*increment from the
#         line_interp[j] = ((N_interps-i)*line1[j-increment*i] + (i)*line2[j+increment*i])/N_interps


line125 = np.zeros(len(line1[:-cutoff_len]))
line150 = np.zeros(len(line1[:-cutoff_len]))
line175 = np.zeros(len(line1[:-cutoff_len]))

for j in range(len(line1)-cutoff_len):
    # Look backwards i*increment from the
    line125[j] = (3*line1[j-1*increment] + 1*line2[j+3*increment])/4
    line150[j] = (2*line1[j-2*increment] + 2*line2[j+2*increment])/4
    line175[j] = (1*line1[j-3*increment] + 3*line2[j+1*increment])/4

print(len(line2))
line1 = line1[:-cutoff_len]
line2 = line2[:-cutoff_len]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize = (15,12))
ax1.plot(range(len(line1)), line1)
ax1.plot(range(len(line125)), line125)

ax2.plot(range(len(line125)), line125)
ax2.plot(range(len(line150)), line150)

ax3.plot(range(len(line150)), line150)
ax3.plot(range(len(line175)), line175)

ax4.plot(range(len(line175)), line175)
ax4.plot(range(len(line2)), line2)
#ax.set_ylim(0,1)
plt.show()

lags = signal.correlation_lags(len(line1), len(line125))
corr1 = signal.correlate(line1, line125)
print(lags[np.argmax(corr1)])

lags = signal.correlation_lags(len(line125), len(line150))
corr2 = signal.correlate(line125, line150)
print(lags[np.argmax(corr2)])

lags = signal.correlation_lags(len(line150), len(line175))
corr3 = signal.correlate(line150, line175)
print(lags[np.argmax(corr3)])

lags = signal.correlation_lags(len(line175), len(line2))
corr4 = signal.correlate(line175, line2)
print(lags[np.argmax(corr4)])


#%% 2D correlation stuff that didn't really work
# imageReconstruct1 = imageReconstruct1[:,:]
# imageReconstruct2 = imageReconstruct2[:,:]

# imageReconstruct125 = np.zeros(np.shape(imageReconstruct1))
# row_range = np.shape(imageReconstruct1)[0]
# col_range = np.shape(imageReconstruct1)[1]

# for i in range(row_range):
#     for j in range(col_range-cutoff_len):
#         imageReconstruct125[i,j] = (3*imageReconstruct1[i, j-1*increment] + 1*imageReconstruct2[i,j+3*increment])/4


# img1 = imageReconstruct1[:, cutoff_len:-cutoff_len]
# img125 = imageReconstruct125[:, cutoff_len:-cutoff_len]
# corr2 = signal.correlate2d(img1, img125, boundary='symm', mode='same')

# print(np.unravel_index(corr2.argmax(), corr2.shape))

# fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1, figsize=(15, 9))
# ax_orig.imshow(img1, cmap='gray')
# ax_orig.set_title('Frame 1')
# ax_orig.axhline(y=row_search, color='r', linestyle='-')
# ax_orig.set_axis_off()

# ax_template.imshow(img125, cmap='gray')
# ax_template.set_title('Frame 2')
# ax_template.axhline(y=row_search, color='r', linestyle='-')
# ax_template.set_axis_off()

# ax_corr.imshow(corr2, cmap = 'gray')
# ax_corr.set_title('2D correlation')
# ax_corr.set_axis_off()

# fig.show()




# Correlate the signals in space and determine the pixel offset
# corr = signal.correlate(line_t0, line_interp, "full")

# Now set the old t0 line to the new line for the next iteration
# line_t0 = np.copy(line_interp)

# # Knowing how far the signal travelled and the sampling rate,
# # we can estimate the speed at which the signal travelled
# WP_convect_pix = np.abs(lags[np.argmax(corr)])
# WP_Up = WP_convect_pix * mm_pix *1e-3 / (dt/N_interps)
# print(WP_convect_pix)
# print(WP_Up)

#



    
