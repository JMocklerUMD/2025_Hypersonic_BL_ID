# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 09:36:37 2025

@author: Joseph Mockler
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import os

#%% Load classification results
WP_io = np.load('True_class.npy')
Confidence_history = np.load('Coonfidence_class.npy')


#%%
print(Confidence_history[10].shape)

#%% read in images
print('Reading training data file')

# Write File Name
file_name = 'C:\\UMD GRADUATE\\RESEARCH\\Hypersonic Image ID\\videos\\Test1\\ConeFlare_Shot64_re33_0deg\\training_data.txt'
if os.path.exists(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len = len(lines)
print(f"{lines_len} lines read")


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

#%% 
def calc_windowed_confid(j, confidence):
    if j == 0:
        local_confid = confidence[j] + confidence[j+1] + confidence[j+2]
    elif j == len(Imagelist) - 1:
        local_confid = confidence[j] + confidence[j-1] + confidence[j-2]
    else:
        local_confid = confidence[j] + confidence[j-1] + confidence[j+1]
        
    return local_confid

def filter_and_classify_frame(Imagelist, confidence, indiv_thres, confid_thres):
    # build out the components of a confusion matrix
    n00, n01, n10, n11 = 0, 0, 0, 0 
    
    filtered_result = []
    for j, _ in enumerate(Imagelist):
        local_confid = calc_windowed_confid(j, confidence)
        
        if (local_confid > confid_thres) or (confidence[j] > indiv_thres):
            filtered_result.append(1)
        else:
            filtered_result.append(0)
            
        # Get stats on the current image
        if WP_io[j] == 0:
            if filtered_result[j] == 0:
                n00 += 1
            if filtered_result[j] == 1:
                n01 += 1 
        elif WP_io[j] == 1:
            if filtered_result[j] == 0:
                n10 += 1
            if filtered_result[j] == 1:
                n11 += 1
                
    return filtered_result, n00, n01, n10, n11

#%% Iterate through the list!
N_img = lines_len
acc_history = []
TP_history = []
TN_history = []
FP_history = []
FN_history = []
WP_io_history = []
confidence_history = []
plot_flag = 0       # View the images? MUCH SLOWER

#thresholds = np.linspace(0, 3, num=50)
thresholds = 1.5
TPRs, FPRs, Pres = [], [], []

for confid_thres in thresholds:
    TP, FP, TN, FN = 0, 0, 0, 0
    for i_iter in range(lines_len):
        
        # Split up image and get the labelled confidence
        Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines)
        confidence = Confidence_history[i_iter]
        
        # Restack and plot the image
        imageReconstruct = np.hstack([image for image in Imagelist])
        
        if plot_flag == 1:
            fig, ax = plt.subplots(1)
            ax.imshow(imageReconstruct, cmap = 'gray')
        
        # Begin filtering the confidence
        #confid_thres = 1.5
        window_size = 3
        indiv_thres = 0.85
        
        filtered_result, n00, n01, n10, n11 = filter_and_classify_frame(Imagelist, confidence, indiv_thres, confid_thres)
        
        if plot_flag == 1:
            for j, _ in enumerate(Imagelist):
                # Add in the classification guess
                if filtered_result[j] == 1:
                    rect = Rectangle((j*slice_width, 5), slice_width, height-10,
                                             linewidth=0.5, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)

            # Check if there's even a bounding box in the image
            if sm_bounds[0] == 'X':
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN class')
                plt.show()
                continue
            else:
                # Add the ground truth over the entire box
                ax.add_patch(Rectangle((sm_bounds[0],sm_bounds[1]), sm_bounds[2], sm_bounds[3], edgecolor='blue', facecolor='none'))
            
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN class')
                plt.show()
                
        # Compute the inter-image accuracy
        #acc = (n00 + n11) / (n00 + n11 + n10 + n01)
        #print(f'Image {i_iter}: accuracy = {acc}')
        
        # Save off data for whole-set analysis
        #TP_history.append(n11)
        #TN_history.append(n00)
        #FP_history.append(n01)
        #FN_history.append(n10)
        #acc_history.append(acc)
        
        # Finally, add to the grand list per threshold
        TP = TP + n11
        FP = FP + n01
        TN = TN + n00
        FN = FN + n10
        
    # Now calculate the percentages
    TPRs.append(TP/(TP+FN))
    FPRs.append(FP/(FP+TN))
    if (TP+FP) == 0:
        Pres.append(1.0)
    else:
        Pres.append(TP/(TP+FP))
    

#%% Create the PR plot
# Plot the curve
fig, (ax, ax2) = plt.subplots(1,2, figsize = (16,8))
ax.plot(FPRs, TPRs, '--.', markersize=10)
ax.plot(np.linspace(0,1,num=100), np.linspace(0,1,num=100))
ax.set_title('ROC Curve')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')

ax2.plot(TPRs, Pres, '--.', markersize=10)
ax2.plot(np.linspace(0,1,num=100), np.flip(np.linspace(0,1,num=100)))
ax2.set_title('Precision-Recall Curve')
ax2.set_xlabel('Recall (True Positive Rate)')
ax2.set_ylabel('Precision')

plt.show()

#%%
# Compute the AUC of the ROC - simple rectangular integration
AUC = 0.0
for i in range(1,len(TPRs)):
    AUC = AUC + (FPRs[i-1]-FPRs[i])*(TPRs[i]+TPRs[i-1])/2    
print(f'Area under the ROC Curve = {AUC}')

PR = 0.0
for i in range(1,len(TPRs)):
    PR = PR + (Pres[i]+Pres[i-1])*(TPRs[i-1]-TPRs[i])/2    
print(f'Area under the PR Curve = {PR}')


#%% Make history plot

# Take a rolling average
def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

Nframe_per_img = len(Imagelist)
n = 20 # Moving avg window
    
fig, (pl1, pl2, pl3, pl4, pl5) = plt.subplots(5,1, figsize = (16,16))
pl1.plot(range(len(acc_history)), acc_history)
pl1.plot(range(n-1, len(acc_history)), moving_average(acc_history, n), color='k', linewidth = 2)
pl1.set_title('Accuracy')

pl2.plot(range(len(TP_history)), [img_stat/Nframe_per_img for img_stat in TP_history])
pl2.plot(range(n-1, len(acc_history)), moving_average(TP_history, n)/Nframe_per_img, color='k', linewidth = 2)
pl2.set_title('True positive rate')

pl3.plot(range(len(TN_history)), [img_stat/Nframe_per_img for img_stat in TN_history])
pl3.plot(range(n-1, len(TN_history)), moving_average(TN_history, n)/Nframe_per_img, color='k', linewidth = 2)
pl3.set_title('True negative rate')

pl4.plot(range(len(FP_history)), [img_stat/Nframe_per_img for img_stat in FP_history])
pl4.plot(range(n-1, len(FP_history)), moving_average(FP_history, n)/Nframe_per_img, color='k', linewidth = 2)
pl4.set_title('False positive rate')

pl5.plot(range(len(FN_history)), [img_stat/Nframe_per_img for img_stat in FN_history])
pl5.plot(range(n-1, len(FN_history)), moving_average(FN_history, n)/Nframe_per_img, color='k', linewidth = 2)
pl5.set_title('False negative rate')

#%% Print out the entire data set statistics
print("Data set statistics")
print("----------------------------------------")
print(f"Whole-set Average: {np.mean(acc_history)}")
print(f"Whole-set True Positive rate: {np.mean(TP_history)/Nframe_per_img}")
print(f"Whole-set True Negative rate: {np.mean(TN_history)/Nframe_per_img}")
print(f"Whole-set False Positive rate: {np.mean(FP_history)/Nframe_per_img}")
print(f"Whole-set False Negative rate: {np.mean(FN_history)/Nframe_per_img}")





















