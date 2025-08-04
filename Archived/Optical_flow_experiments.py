# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:22:15 2025

@author: Joseph Mockler
"""

import numpy as np
import cv2 as cv
import argparse
import os
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.patches import Rectangle


#%% 

# Read 2 frames

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

def perform_signal_recreation(i1, i2, lines):
    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i1, lines)
    frame1 = np.hstack(Imagelist)

    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i2, lines)
    frame2 = np.hstack(Imagelist)

    mm_pix = 0.0756         # From paper
    FR = 285e3              # Camera frame rate in Hz
    dt = 1/FR               # time step between frames
    prop_speed = 815        # A priori estimate of propogation speed


    pix_tr = prop_speed * dt * 1/(mm_pix*1e-3)  # Computed number of pixels traveled between frames
    row_range = frame1.shape[0]
    col_range = frame1.shape[1]

    N_interps = 4
    increment = round(pix_tr/(N_interps+1))
    cutoff_len = round(pix_tr)
    # Our recreated signals should chop off the front and back bit to avoid
    # any weird clipping I noticed
    recreated_signals = np.zeros((N_interps, row_range, col_range))
    # Bandpass filter the frames
    frame1 = bandpass_the_frame(frame1)
    frame2 = bandpass_the_frame(frame2)

    # ii forms the intermediate frames
    for ii in range(1,N_interps+1):
        
        # j iterates over the cols (weighted avg is equally applied across the entire col)
        for j in range(cutoff_len, col_range-cutoff_len):
            # Look backwards i*increment from the
            step_bwd = ii*increment
            step_fwd = ((N_interps+1)-ii)*increment
            recreated_signals[ii-1,:,j] = (((N_interps+1)-ii)*frame1[:,j-step_bwd] + ii*frame2[:,j+step_fwd])/(N_interps+1)

    return recreated_signals

def bandpass_the_frame(frame1):
    sos = signal.butter(4, [0.0133, 0.10], btype='bandpass', fs=1, output='sos')
    row_range = frame1.shape[0]
    col_range = frame1.shape[1]

    for i in range(row_range):
        frame1[i,:] = signal.sosfiltfilt(sos, frame1[i,:])

    return frame1

def convert_frame_to_8bit(frame1):
    return np.uint8(cv.normalize(frame1, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX))
    


#%%
file_name = 'C:\\UMD GRADUATE\\RESEARCH\\Hypersonic Image ID\\videos\\Test1\\run34\\full_video_data.txt'
if os.path.exists(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len = len(lines)
print(f"{lines_len} lines read")

#%%
# Pick out two frames with WP's

for ii in range(500, 600):
    fig, ax = plt.subplots(1)
    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(ii, lines)
    imageReconstruct = np.hstack(Imagelist)
    
    ax.imshow(imageReconstruct, cmap='gray')
    ax.set_title('Frame '+str(ii))
    plt.show()
    

#%% Analyze frames 561 and 562

i1, i2 = 561, 562

Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i1, lines)
frame1 = np.hstack(Imagelist)

Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i2, lines)
frame2 = np.hstack(Imagelist)

sos = signal.butter(4, [0.0133, 0.10], btype='bandpass', fs=1, output='sos')
row_range = frame1.shape[0]
col_range = frame1.shape[1]

for i in range(row_range):
    frame1[i,:] = signal.sosfiltfilt(sos, frame1[i,:])
    frame2[i,:] = signal.sosfiltfilt(sos, frame2[i,:])

frame1 = np.uint8(cv.normalize(frame1, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX))
frame2 = np.uint8(cv.normalize(frame2, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX))

#frame1 = cv.Canny(frame1, 100, 200, 1)
#frame2 = cv.Canny(frame2, 100, 200, 1)

#frame1 = frame1[30:64, 400:700]
#frame2 = frame2[30:64, 400:700]

# frame1 = frame1[:,200:700]
# frame2 = frame2[:,200:700]

#frame1 = cv.equalizeHist(frame1)
#frame2 = cv.equalizeHist(frame2)

offset = 10
start_col, stop_col = 200, 800
start_row, stop_row = 0, 64

frame1 = frame1[start_row:stop_row, start_col:stop_col]
frame2 = frame2[start_row:stop_row, start_col+offset:stop_col+offset]

# now try bringing frame 2 back by 30 pixels
frame2[:,start_col:stop_col] = frame2[:,start_col:stop_col+offset]
frame2 = np.delete(frame2, slice(stop_col, stop_col+offset), axis = 1)
#frame1 = frame1[:,0:400]


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,2))
ax1.imshow(frame1)
#ax1.axhline(y=55, color='r', linestyle='-')
ax1.set_title('Frame '+str(i1))
ax2.imshow(frame2)
ax2.set_title('Frame '+str(i2))
#rect = Rectangle((225,40), 200,15,
#                         linewidth=1, edgecolor='red', facecolor='none')
#ax2.add_patch(rect)
plt.show()

# fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,2))
# ax1.imshow(frame1)

# ax2.imshow(frame2)

# plt.show()

#%%
# Pass thru Farneback's implementation of optical flow
# flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 2, 5, 3, 7, 1.1, 0)
u = flow[:,:,0]
v = flow[:,:,1]

fig, axes = plt.subplots(2,1, figsize=(10,2))

minmin = np.min([np.min(u), np.min(v)])
maxmax = np.max([np.max(u), np.max(v)])

im1 = axes[0].imshow(u, vmin=1.1*np.min(u), vmax=0.8*np.max(u))
im2 = axes[1].imshow(v, vmin=1.1*np.min(v), vmax=0.8*np.max(v))
cbar_ax = fig.add_axes([0.92, 0.1, 0.04, 0.7])
fig.colorbar(im1, cax=cbar_ax)

cbar_ax = fig.add_axes([1.0, 0.1, 0.04, 0.7])
fig.colorbar(im2, cax=cbar_ax)

axes[0].set_title('Farneback Optical Flow Results')
plt.show()

#%% Test out making a displacement vs offset plot
displacements = []
measured_prop = []
time_vec = []
start_col, stop_col = 200, 800
start_row, stop_row = 0, 64
import time 

st_time = time.time()
for offset in range(80):
    
    i1, i2 = 561, 562

    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i1, lines)
    frame1 = np.hstack(Imagelist)

    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i2, lines)
    frame2 = np.hstack(Imagelist)

    sos = signal.butter(4, [0.0133, 0.10], btype='bandpass', fs=1, output='sos')
    row_range = frame1.shape[0]
    col_range = frame1.shape[1]
    
    st_time = time.time()
    for i in range(row_range):
        frame1[i,:] = signal.sosfiltfilt(sos, frame1[i,:])
        frame2[i,:] = signal.sosfiltfilt(sos, frame2[i,:])

    frame1 = np.uint8(cv.normalize(frame1, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX))
    frame2 = np.uint8(cv.normalize(frame2, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX))
    
    frame1 = frame1[start_row:stop_row, start_col:stop_col]
    frame2 = frame2[start_row:stop_row, start_col+offset:stop_col+offset]

    # now try bringing frame 2 back by 30 pixels
    frame2[:,start_col:stop_col] = frame2[:,start_col:stop_col+offset]
    frame2 = np.delete(frame2, slice(stop_col, stop_col+offset), axis = 1)
    
    flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 2, 5, 3, 7, 1.1, 0)
    u = flow[:,:,0]
    v = flow[:,:,1]
    end_time = time.time()
    time_vec.append(end_time-st_time)
    
    displacements.append(np.median(u[30:60,100:500]))
    measured_prop.append(displacements[offset] + offset)
    

#%%
print(f'Mean compute time: {np.mean(time_vec)} sec')
fig, ax = plt.subplots(1, figsize = (7,5))
ax.set_title('Computed Prop Speed vs 2nd Frame Offset')
ax.set_xlabel('2nd frame offset upstream, pixels')
ax.set_ylabel('Median Disp. + Offset, pixles')
ax.plot(range(80), displacements, 'k.')
plt.show()
print(f'Mean prop in linear region = {np.mean(measured_prop[25:47])}')

#%%
lin_fit = np.polyfit(range(25,45), displacements[25:45], 1)
x_lin_plot = np.linspace(25, 45, 100)
y_lin_plot = lin_fit[0]*x_lin_plot + lin_fit[1]

est_prop = lin_fit[1]/lin_fit[0]

ax.plot(x_lin_plot, y_lin_plot, 'r--')
print(f'Est prop speed: {est_prop} pixels')

mm_pix = 0.0756         # From paper
FR = 285e3              # Camera frame rate in Hz
dt = 1/FR               # time step between frames

print(f'Est prop speed: {est_prop*mm_pix*1e-3/dt} m/s')

plt.show()

#%% Set of code to demonstrate optical flow using the Farneback Dense method
# Demonstrates the algorithm on just two frames and on a local-in-time recreated
# signal that interpolates the two frames

# -----------------------------------------
# I. Demonstrate on two individual frames
i1, i2 = 561, 562
ColStart, ColEnd = 450, 625 # Pick bounds to analyze

# Pick out the first two frames
Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i1, lines)
frame1 = np.hstack(Imagelist)

Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i2, lines)
frame2 = np.hstack(Imagelist)

# Take subset of the frame that we want to analyze
frame1 = frame1[:,ColStart:ColEnd]
frame2 = frame2[:,ColStart:ColEnd]

# Bandpass filter the frames
frame1 = bandpass_the_frame(frame1)
frame2 = bandpass_the_frame(frame2)

# Now convert to 8bit - leaving in the normalized [0,1] 64 float gives junk results
frame1 = convert_frame_to_8bit(frame1)
frame2 = convert_frame_to_8bit(frame2)

# Now compute the dense flow at each point in the frame
# flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 5, 3, 7, 1.1, 0)
u = flow[:,:,0] # X components
v = flow[:,:,1] # Y components

# Make a big plot! Top frame is x, bottom frame is y
fig, axes = plt.subplots(2,1, figsize=(10,2))
im1 = axes[0].imshow(u, vmin=1.1*np.min(u), vmax=0.8*np.max(u))
im2 = axes[1].imshow(v, vmin=1.1*np.min(v), vmax=0.8*np.max(v))
cbar_ax = fig.add_axes([0.65, 0.1, 0.04, 0.7])
fig.colorbar(im1, cax=cbar_ax)

cbar_ax = fig.add_axes([0.75, 0.1, 0.04, 0.7])
fig.colorbar(im2, cax=cbar_ax)

axes[0].set_title('Farneback Optical Flow Results: Original')
plt.show()


# -----------------------------------------
# II. Use recreated signals example

# First perform the weighted-in-time signal recreation
recreated_signals = perform_signal_recreation(i1, i2, lines)

# Pick two of the signals to compare and convert them to 8-bit
frame1, frame2 = recreated_signals[1,:,:], recreated_signals[2,:,:]
frame1, frame2 = frame1[:,ColStart:ColEnd], frame2[:,ColStart:ColEnd]

frame1 = convert_frame_to_8bit(frame1)
frame2 = convert_frame_to_8bit(frame2)

# Now compute the dense flow at each point in the frame
# flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 5, 2, 3, 1, 1.5, 0)
u = flow[:,:,0] # X components
v = flow[:,:,1] # Y components

# Make a big plot! Top frame is x, bottom frame is y
fig, axes = plt.subplots(2,1, figsize=(10,2))
im1 = axes[0].imshow(u, vmin=1.1*np.min(u), vmax=0.8*np.max(u))
im2 = axes[1].imshow(v, vmin=1.1*np.min(v), vmax=0.8*np.max(v))
cbar_ax = fig.add_axes([0.65, 0.1, 0.04, 0.7])
fig.colorbar(im1, cax=cbar_ax)

cbar_ax = fig.add_axes([0.75, 0.1, 0.04, 0.7])
fig.colorbar(im2, cax=cbar_ax)

axes[0].set_title('Farneback Optical Flow Results: Recreated')
plt.show()

#%% Set of code to demonstrate optical flow using the LK method
# Demonstrates the algorithm on just two frames and on a local-in-time recreated
# signal that interpolates the two frames

# -----------------------------------------
# I. Demonstrate on two individual frames
i1, i2 = 561, 562
ColStart, ColEnd = 450, 625 # Pick bounds to analyze

# Pick out the first two frames
Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i1, lines)
frame1 = np.hstack(Imagelist)

Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i2, lines)
frame2 = np.hstack(Imagelist)

# Take subset of the frame that we want to analyze
frame1 = frame1[:,ColStart:ColEnd]
frame2 = frame2[:,ColStart:ColEnd]

# Bandpass filter the frames
frame1 = bandpass_the_frame(frame1)
frame2 = bandpass_the_frame(frame2)

# Now convert the frame to an 8-bit frame for cohesion w/ the opencv algorithm
frame1 = convert_frame_to_8bit(frame1)
frame2 = convert_frame_to_8bit(frame2)

# Take the Canny to get contour points
frame1_canny = cv.Canny(frame1, 100, 300, 1)

# Now get the contour
contours, heirarchy = cv.findContours(frame1_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
p0 = np.float32(contours[2]) # Take a small subset of contour points

# Plot the Canny edge detection results and a few
fig, ax1 = plt.subplots(1, figsize=(10,2))
ax1.imshow(frame1_canny)
ax1.plot(p0[:,0,0], p0[:,0,1], '.r', markersize=1.5)
ax1.set_title('Frame '+str(i1)+ ' Canny Edge Detection')
plt.show()

# Now let's perform the analysis - first set some parameters
lk_params = dict( winSize  = (5, 5),
                  maxLevel = 3,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Then, for all the detected contour points, run the optical flow algorithm
u, v = [], []
for contour in contours:
    p1, st, err = cv.calcOpticalFlowPyrLK(frame1.copy(), frame2.copy(), np.float32(contour), None, **lk_params)
    for i in range(p1.shape[0]):
        u.append(p1[i,0,0]-contour[i,0,0])
        v.append(p1[i,0,1]-contour[i,0,1])

# Print the results
print(f'Flow velocity w/ original = [{np.mean(u)}, {np.mean(v)}] +/- [{np.std(u)}, {np.std(v)}]')

# And finally, make a plot to show the movement of just a few contour points
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,4))
ax1.imshow(frame1)
ax1.plot(p0[:,0,0], p0[:,0,1], '.r', markersize=1.5)
ax1.set_title('Frame '+str(i1)+': Original Frames')
ax2.imshow(frame2)
p1, st, err = cv.calcOpticalFlowPyrLK(frame1.copy(), frame2.copy(), p0, None, **lk_params)
ax2.plot(p1[:,0,0], p0[:,0,1], '.r', markersize=1.5)
plt.show()

# -----------------------------------------
# II. Use recreated signals example

# First perform the weighted-in-time signal recreation
recreated_signals = perform_signal_recreation(i1, i2, lines)

# Pick two of the signals to compare and convert them to 8-bit
frame1, frame2 = recreated_signals[1,:,:], recreated_signals[2,:,:]
frame1, frame2 = frame1[:,ColStart:ColEnd], frame2[:,ColStart:ColEnd]

frame1 = convert_frame_to_8bit(frame1)
frame2 = convert_frame_to_8bit(frame2)

# Take the canny and get the contours
frame1_canny = cv.Canny(frame1, 100, 300, 1)
contours, heirarchy = cv.findContours(frame1_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
p0 = np.float32(contours[2])

fig, ax1 = plt.subplots(1, figsize=(10,2))
ax1.imshow(frame1_canny)
ax1.plot(p0[:,0,0], p0[:,0,1], '.r', markersize=1.5)
ax1.set_title('Frame '+str(i1)+ ' Canny Edge Detection')
plt.show()

# Now perform the optical flow over all detected contour points
# Then, for all the detected contour points, run the optical flow algorithm
u, v = [], []
for contour in contours:
    p1, st, err = cv.calcOpticalFlowPyrLK(frame1.copy(), frame2.copy(), np.float32(contour), None, **lk_params)
    for i in range(p1.shape[0]):
        u.append(p1[i,0,0]-contour[i,0,0])
        v.append(p1[i,0,1]-contour[i,0,1])

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,4))
ax1.imshow(frame1)
ax1.plot(p0[:,0,0], p0[:,0,1], '.r', markersize=1.5)
#ax1.axhline(y=48, color='r', linestyle='-')
ax1.set_title('Frame '+str(i1)+': Recreated Signals')
ax2.imshow(frame2)
plt.show()

print(f'Flow velocity w/ recreated = [{np.mean(u)}, {np.mean(v)}] +/- [{np.std(u)}, {np.std(v)}]')
       

#%% Run the script to see all the filtered frame, stacked, with Farneback
# First calculate approx how many pixels a wave will propogate in a single frame
N_interps = 4
i1, i2 = 561, 562
ColStart, ColEnd = 450, 625 # Pick bounds to analyze

# Pick out the first two frames
Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i1, lines)
frame1 = np.hstack(Imagelist)
Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i2, lines)
frame2 = np.hstack(Imagelist)

# Bandpass filter the frames
frame1 = bandpass_the_frame(frame1)
frame2 = bandpass_the_frame(frame2)

recreated_signals = perform_signal_recreation(i1, i2, lines)
recreated_signals = recreated_signals[:,30:64, 200:700]
frame1 = frame1[30:64, 200:700]
frame2 = frame2[30:64, 200:700]

fig, axes = plt.subplots(N_interps+2, 1, figsize=(10,2*(N_interps+1)))
for ii in range(N_interps+2):
    if ii == 0:
        axes[ii].imshow(frame1)
    elif ii == N_interps+1:
        axes[ii].imshow(frame2)
    else:
        axes[ii].imshow(recreated_signals[ii-2,:,:])
plt.show()

# flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
fig_u, axes_u = plt.subplots(N_interps+1,1, figsize=(10,2*N_interps))
fig_v, axes_v = plt.subplots(N_interps+1,1, figsize=(10,2*N_interps))
for ii in range(N_interps+1):
    if ii == 0: # Frame 1 to recreated signal 1
        frame_ii = np.uint8(cv.normalize(frame1, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX))
        frame_iip1 = np.uint8(cv.normalize(recreated_signals[0,:,:], None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX))
    elif ii == N_interps: # Intermediate recreated signals
        frame_ii = np.uint8(cv.normalize(recreated_signals[ii-1,:,:], None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX))
        frame_iip1 = np.uint8(cv.normalize(frame2, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX))
    else: # Last recreated signal to frame 2
        frame_ii = np.uint8(cv.normalize(recreated_signals[ii-1,:,:], None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX))
        frame_iip1 = np.uint8(cv.normalize(recreated_signals[ii,:,:], None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX))
    
    flow = cv.calcOpticalFlowFarneback(frame_ii, frame_iip1, None, 0.5, 3, 5, 3, 7, 1.1, 0)
    u = flow[:,:,0]
    v = flow[:,:,1]

    im_u = axes_u[ii].imshow(u)
    cbar_ax = fig_u.add_axes([0.25/2, -0.2*ii, 0.75, 0.04])
    fig_u.colorbar(im_u, cax=cbar_ax, orientation='horizontal')
    
    im_v = axes_v[ii].imshow(v)
    cbar_ax = fig_v.add_axes([0.25/2, -0.2*ii, 0.75, 0.04])
    fig_v.colorbar(im_v, cax=cbar_ax, orientation='horizontal')
    
plt.show()


#%%

cap = cv.VideoCapture('C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\slow_traffic_small.mp4')
 
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
 
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
 
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
 
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
 
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
i = 0
while i < 100:
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
 
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
 
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
 
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)
 
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
 
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    
    i += 1
    
    
cv.destroyAllWindows()

#%%

cap = cv.VideoCapture('C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\slow_traffic_small.mp4')
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
 
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('frame2', bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr)
    prvs = next
 
cv.destroyAllWindows()