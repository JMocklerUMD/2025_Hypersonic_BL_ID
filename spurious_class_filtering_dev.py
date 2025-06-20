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
WP_io = np.load('C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\True_class_Run38.npy')
Confidence_history = np.load('C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\Confidence_class_Run38.npy')


#%% read in images
print('Reading training data file')

# Write File Name
file_name = 'C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\training_data_LangleyRun38.txt'
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

#%% Windowing and filtering functions
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

def filter_and_classify_frame(Imagelist, confidence, WP_io, indiv_thres, confid_thres, window_size):
    '''
    Classifies all the slices in a single frame using the windowing method. 
    Compares to a pre-defined threshold to determine if the a 2nd mode wave 
    packet is likely present or not.
    '''
    
    # build out the components of a confusion matrix
    n00, n01, n10, n11 = 0, 0, 0, 0 
    
    filtered_result = []
    for j, _ in enumerate(Imagelist):
        local_confid = calc_windowed_confid(j, confidence, window_size)
        
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

def filter_by_simple_threshold(Imagelist, confidence, WP_io, confid_thres):
    n00, n01, n10, n11 = 0, 0, 0, 0 
    filtered_result = []
    for j, _ in enumerate(Imagelist):
        if (confidence[j] > confid_thres):
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

#%% Create the FP/FN vs threshold plots
TPRs, FPRs = [], []
TP_count, FP_count, TN_count, FN_count = [], [], [], []
thresholds = np.linspace(0,3,20)
for confid_thres in thresholds:  
    TP, FP, TN, FN = 0, 0, 0, 0
    for i_iter in range(lines_len):
        
        # Split up image and get the labelled confidence
        Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines)
        confidence = Confidence_history[i_iter]
        
        # Begin filtering the confidence
        # confid_thres = 1.5
        window_size = 3
        indiv_thres = 0.85
        
        # Perform the windowing and classify the frame
        filtered_result, n00, n01, n10, n11 = filter_and_classify_frame(Imagelist, confidence, WP_io, indiv_thres, confid_thres, window_size)
        
        # Finally, add to the grand list per threshold
        TP = TP + n11
        FP = FP + n01
        TN = TN + n00
        FN = FN + n10
        
    # At the end of each threshold, append to the master list
    TP_count.append(TP)
    FP_count.append(FP)
    TN_count.append(TN)
    FN_count.append(FN)
    TPRs.append(TP/(TP+FN))
    FPRs.append(FP/(FP+TN))
    
#%% Create a big plot of the results from above 
# Creates the trend plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (16,8))
ax1.plot(thresholds, TP_count, '--.', markersize=10)
ax1.set_ylabel('True Positive Count')
ax1.set_ylim(0, 250)
ax2.plot(thresholds, TN_count, '--.', markersize=10)
ax2.set_ylabel('True Negative Count')
ax2.set_ylim(0, 7000)
ax3.plot(thresholds, FP_count, '--.', markersize=10)
ax3.set_xlabel('Windowind threshold')
ax3.set_ylabel('False Positive Count')
ax3.set_ylim(0, 7000)
ax4.plot(thresholds, FN_count, '--.', markersize=10)
ax4.set_xlabel('Windowind threshold')
ax4.set_ylabel('False Negative Count')
ax4.set_ylim(0, 250)
fig.suptitle("Analysis of Slice Classifications for Windowing - CF Re33", fontsize=16)
plt.show()

# Creates the response rates plots
fig, ax = plt.subplots(1, figsize=(8,6))
ax.plot(thresholds, TPRs, '--.', markersize = 7, label='True Positive Rate (TP/(TP+FN))')
ax.plot(thresholds, FPRs, '--.', markersize = 7, label='False Positive Rate (FP/(FP+TN))')
ax.set_xlabel('Window Threshold')
ax.set_ylabel('Percent')
ax.legend()
plt.show()
    
#%% Create the data for a big ROC and PR plot comparison
Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(1, lines)
N_img = lines_len
plot_flag = 1               # View the images? MUCH SLOWER
window_size = 3             # Size to perform moving sum over
indiv_thres = 0.85          # Individual exception threshold to meet
N_slices = len(Imagelist)   # Number of slices in frame

# Set a range of thresholds to build ROC, PR curves or other comparisons
# thresholds = np.linspace(0, 3, num=50)

# Set just a 1x1 array for the results of a single threshold
thresholds = [1.5]

filtered_history = []
TPRs, FPRs, Pres = [], [], []
for confid_thres in thresholds:
    TP, FP, TN, FN = 0, 0, 0, 0
    TP_history, TN_history, FP_history, FN_history, acc_history = [], [], [], [], []
    for i_iter in range(N_img): 
        
        # Split up image and get the labelled confidence
        Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines)
        confidence = Confidence_history[i_iter]
        
        # Restack and plot the image
        imageReconstruct = np.hstack([image for image in Imagelist])
        
        if plot_flag == 1:
            fig, ax = plt.subplots(1)
            ax.imshow(imageReconstruct, cmap = 'gray')
        
        # Filter the images by confidence
        filtered_result, n00, n01, n10, n11 = filter_and_classify_frame(Imagelist, confidence, WP_io, indiv_thres, confid_thres, window_size)
        #filtered_result, n00, n01, n10, n11 = filter_by_simple_threshold(Imagelist, confidence, WP_io, confid_thres)
        
        filtered_history.append(filtered_result)
    
        # Overlays the figures for inspection
        if plot_flag == 1:
            for j, _ in enumerate(Imagelist):
                # Add in the classification guess
                if filtered_result[j] == 1:
                    rect = Rectangle((j*slice_width, 5), slice_width, height-10,
                                             linewidth=0.5, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                
                # Adds a rectangle for the confidence of classification at every square
                prob = confidence[j,0]
                #rect = Rectangle((j*slice_width, 5), slice_width, height-10,
                #linewidth=1.0*prob*prob, edgecolor='red', facecolor='none')
                #ax.add_patch(rect)
                ax.text(j*slice_width, height+60,round(prob,2), fontsize = 7)
            
            # Check if there's even a bounding box in the image
            if sm_bounds[0] == 'X':
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN class')
                plt.show()
                #continue
            else:
                # Add the ground truth over the entire box
                ax.add_patch(Rectangle((sm_bounds[0],sm_bounds[1]), sm_bounds[2], sm_bounds[3], edgecolor='blue', facecolor='none'))
            
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN class')
                plt.show()
                
        # Compute the inter-image accuracy
        acc = (n00 + n11) / (n00 + n11 + n10 + n01)
        #print(f'Image {i_iter}: accuracy = {acc}')
        
        # Save off data for whole-set analysis
        TP_history.append(n11)
        TN_history.append(n00)
        FP_history.append(n01)
        FN_history.append(n10)
        acc_history.append(acc)
        
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
        
    print('Completed a threshold!')


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

# Compute the AUC of the ROC - simple rectangular integration
AUC = 0.0
for i in range(1,len(TPRs)):
    AUC = AUC + (FPRs[i-1]-FPRs[i])*(TPRs[i]+TPRs[i-1])/2    
print(f'Area under the ROC Curve = {AUC}')

PR = 0.0
for i in range(1,len(TPRs)):
    PR = PR + (Pres[i]+Pres[i-1])*(TPRs[i-1]-TPRs[i])/2    
print(f'Area under the PR Curve = {PR}')

#%% Save the data to create big comparison plot

# Nominal case
#FRP_nom = FPRs
#TPR_nom = TPRs
#Pres_nom = Pres

# Case with 3x window
#FRP_window = FPRs
#TPR_window = TPRs
#Pres_window = Pres

# Case with 3x window an 85% nominal confidence
#FRP_window_indiv = FPRs
#TPR_window_indiv = TPRs
#Pres_window_indiv = Pres

#%% Create a big ROC/PR Comparison plot
#fig, (ax, ax2) = plt.subplots(1,2, figsize = (16,8))
#ax.plot(FRP_nom, TPR_nom, '--.', markersize=10, label='Nominal')
#ax.plot(FRP_window, TPR_window, '--.', markersize=10, label='Windowed')
#ax.plot(FRP_window_indiv, TPR_window_indiv, '--.', markersize=10, label='Windowed + Indiv.')
#ax.plot(np.linspace(0,1,num=100), np.linspace(0,1,num=100), label = 'Random Classifier')
#ax.set_title('ROC Curve')
#ax.set_xlabel('False Positive Rate')
#ax.set_ylabel('True Positive Rate')
#ax.legend(loc='lower right')

#ax2.plot(TPR_nom, Pres_nom, '--.', markersize=10, label='Nominal')
#ax2.plot(TPR_window, Pres_window, '--.', markersize=10, label='Windowed')
#ax2.plot(TPR_window_indiv, Pres_window_indiv, '--.', markersize=10, label='Windowed + Indiv.')
#ax2.plot(np.linspace(0,1,num=100), np.flip(np.linspace(0,1,num=100)), label = 'Prior Classifier')
#ax2.set_title('Precision-Recall Curve')
#ax2.set_xlabel('Recall (True Positive Rate)')
#ax2.set_ylabel('Precision')
#ax2.legend(loc='lower left')
#plt.show()


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


#%% Create video statistics plots
counts_per_slice = np.zeros(len(Imagelist))
counts_in_time = np.zeros(lines_len)

for i_iter in range(lines_len):
    classes = filtered_history[i_iter]
    for j in range(N_slices):
        counts_per_slice[j] += classes[j]
    
    counts_in_time[i_iter] = sum(classes)


fig, ax = plt.subplots()
ax.grid(zorder=0)
ax.bar(range(N_slices), counts_per_slice, zorder = 3)
ax.set_xlabel("Image slice along direction of flow")
ax.set_ylabel("Summed 2nd mode wave packet counts")
plt.show()


fig, ax = plt.subplots()
ax.plot(range(lines_len), counts_in_time/len(Imagelist))
ax.set_ylabel("% of frame with 2nd mode WP in time")
ax.set_xlabel("Frame number")
plt.show()
















