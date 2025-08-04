"""
DATE:           Mon July 21 10:27:17 2025
AUTHOR:         Joseph Mockler
DESCRPITION:    This program accepts a trained NN model and an unlabeled txt file
                of video data (as generated from the MATLAB script) and outputs
                an array of arrays, which each contain the classification results
                per slice of the frame in the text file. This can be fed into the
                WP_time_tracking script for analysis of the video data (or any
                analyses that may be performed!). Of course, if you have labeled
                text data as well, then this will classify and compare to form 
                performance plots. 
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import os

import keras
import tensorflow as tf

from keras.applications import resnet50
from keras.models import Model

#from keras.preprocessing import image
#from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

# Update to project folder location
base_folder = 'C:\\Users\\tyler\\Desktop\\NSSSIP25\\Machine Learning Classification - NSSSIP25'

os.chdir(base_folder)

from scipy import signal

# Import custom functions
from ML_utils import *
from Results_utils import *
from ML_models import *

#%% Read image and existing classification model into the workspace
print('Reading training data file')

# Are we classifying second-mode wave packets or turbulence?
second_mode = True
turb = False

# Write File Name - set of images to be classified
file_name = f"{base_folder}\\Example Data and Outputs\\video_data_LangleyRun34_105_116ms.txt"
lines, lines_len = write_text_to_code(file_name)

# Transfer learning model for stacking on ResNet50
model1 = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
output = model1.output
output = tf.keras.layers.Flatten()(output)
resnet_model = Model(model1.input,output)

# Load the classifier model
model = keras.models.load_model(f'{base_folder}\\Example Data and Outputs\\Build_Classifier\\secondmodemodel_LangleyRun34.keras')

# File location for saving confidence history results - used for propagation speed calculations
file_save = f'{base_folder}\\Example Data and Outputs\\Classify_a_video\\TESTclassification_results_LangleyRun34_filtered.npy'


plot_flag = 1               # View the images? - slower


window_size = 3             # Moving window to filter the frames
indiv_thres = 0.85          # Individual exception threshold
confid_thres = 1.5          # SUMMED confidence over the entire window. 
                            # e.g. for 0.5 over 3 windows, make this value 1.5
use_post_process = 1        # 1 to use windowing post process, 0 if not
slice_width = 64

bandpass = True             # whether or not to bandpass the input images
BL_height = 19              # boundary layer in pixels
                            # use BL_find in BL Normalization Codes

verbosity = 1               # 1 to get two progress bars for each classified image, 0 if not
                            # Print-outs can help guage how fast the code is running
                            # Too many print-outs can fill up the console and result in the loss of previous print-outs
                            # "Done resizing" will still display for every image
if second_mode and turb:                      
    raise ValueError('This code cannot classify both second-mode wave packets and turbulence. Please use Classify_a_video_SMandTurb.')

#%% Iterate through the list!
# Initialize some arrays for later data analysis
N_img = lines_len
WP_io_history = []
confidence_history = []
filtered_result_history = []
video_statistics = []
frame_res_history = []

cut_low = 1/BL_height
cut_high = 1/(4*BL_height)

sos = signal.butter(4,[cut_high,cut_low],btype='bandpass',fs=1,output='sos')

### Iterate over all frames in the video
for i_iter in range(N_img):
    
    # Repeat essentially the same procedure of image segmentation -> feature extraction ->
    # prediction -> post-process for cohesion in space. Then, plot the frames!
    
    # Read off the image data from the labeled text file and divide into slices
    FrameList, Frame_WP_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines, slice_width)
    
    # Bandpass filter the images (if desired)
    if bandpass:
        for n,slc in enumerate(FrameList):
            for m in range(64):
                slc[m,:] = signal.sosfiltfilt(sos,slc[m,:])
    
    # Resizes the arrays for ResNet50 compatability  
    FrameList_RS = resize_frames_for_ResNet50(FrameList, Frame_WP_io)
    
    # Get the features
    FrameFeatures = get_bottleneck_features(resnet_model, FrameList_RS,  verbose=verbosity)
    
    # Calculate the confidence
    confidence = model.predict(FrameFeatures, verbose=verbosity)
    
    # Apply spatial cohesion to the confidence results
    frame_res, n11, n10, n01, n00 = classify_WP_frame(FrameList, Frame_WP_io, confidence, window_size, indiv_thres, confid_thres, use_post_process)
    
    # Append statistics for whole-video analysis
    video_statistics.append(ClassificationStatistics(n11, n10, n01, n00, len(frame_res)))
    
    # Save off some statistics for building a ROC and PR curve later!
    confidence_history.append(confidence)
    WP_io_history.append(Frame_WP_io)
    
    # Save off post-processed results for later analysis (ex: intermittency plot) if desired
    frame_res_history.append(frame_res)
    
    # Plot the frame
    if plot_flag == 1:
        if second_mode: 
            fig, ax = plot_frame(FrameList, frame_res, confidence, slice_width, 1, 0)
        if turb:
            fig, ax = plot_frame(FrameList, frame_res, confidence, slice_width, 0, 1)
    
        # Check if there's even a bounding box in the image
        if sm_bounds[0] == 'X':
            if second_mode:
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN WP')
            if turb:
                ax.set_title('Image '+str(i_iter)+'. Blue: true Turb. Red: NN Turb')
            plt.show()
            continue
        else:
            # Add the ground truth over the entire box
            ax.add_patch(Rectangle((sm_bounds[0], sm_bounds[1]), sm_bounds[2], sm_bounds[3], edgecolor='blue', facecolor='none'))
            if second_mode:
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN WP')
            if turb:
                ax.set_title('Image '+str(i_iter)+'. Blue: true Turb. Red: NN Turb')
            plt.show()
    
print('Done classifying the video!')

#%% Save the classification results for prop speed calcs
np.save(file_save, confidence_history)
print('Confidence history array saved')

#%% Now make a history plot of the results using the ClassificationStatistics class
if np.sum(WP_io_history) == 0:
    print('Set appears to be unlabeled. Stats not run.')
else:
    N_img = len(video_statistics)
    acc_history = [Stats.acc for Stats in video_statistics]
    TP_history = [Stats.TP for Stats in video_statistics]
    TN_history = [Stats.TN for Stats in video_statistics]
    FP_history = [Stats.FP for Stats in video_statistics]
    FN_history = [Stats.FN for Stats in video_statistics]
    
    Nframe_per_img = len(FrameList)
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
    plt.show()
    
    # This first function grabs the FPR, TPR, and Pres, which you need to construct the ROC curves
    FPRs, TPRs, Pres = form_ROC_PR_statistics(FrameList, WP_io_history, confidence_history, window_size, use_post_process)
    
    # If you just want to plot the curves, call the fcn below.
    AUC, PR, fig, ax, ax2 = form_ROC_Curve(FrameList, WP_io_history, confidence_history, window_size, use_post_process)
