"""
DATE:           Mon July 21 10:27:17 2025
AUTHOR:         Joseph Mockler
DESCRPITION:    This program accepts two trained NN models and an unlabeled txt file
                of video data (as generated from the MATLAB script) and outputs
                two arrays of arrays, which each contain the classification results
                per slice of the frame in the text file, one for second-mode wave
                packets and one for turbulence. This can be fed into the
                functions in the Turbulence_and_breakdown_utils code for further analysis.
                It also outputs an array of arrays, which each contain the wave packet classification 
                results per slice of the frame in the text file. This can be fed into the
                WP_time_tracking script for analysis of the video data (or any
                analyses that may be performed!). Of course, if you have labeled text 
                data as well, then this will classify and compare to form performance 
                plots for wave packets and turbulence individually, but these will only
                be accurate for which ever feature is labeled in the video data.
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

# Import custom functions
from ML_utils import *
from Results_utils import *
from ML_models import *

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

from scipy import signal

import copy


#%% Read image and existing classification model into the workspace
print('Reading training data file')

# Write File Name
model_path_wp = f'{base_folder}\\Example Data and Outputs\\Build_Classifier\\secondmodemodel_LangleyRun34.keras'
model_path_turb = f'{base_folder}\\Example Data and Outputs\\Build_Classifier\\turbulencemodel_LangleyRun34.keras'
file_name = f"{base_folder}\\Example Data and Outputs\\video_data_LangleyRun34_105_116ms.txt"
lines, lines_len = write_text_to_code(file_name)

# File paths for saving data
file_save_wp = f'{base_folder}\\Example Data and Outputs\\Classify_a_video_SMandTurb\\TESTwp_result_history.npy'
file_save_turb = f'{base_folder}\\Example Data and Outputs\\Classify_a_video_SMandTurb\\TESTturb_result_history.npy'
file_save_confid = f'{base_folder}\\Example Data and Outputs\\Classify_a_video_SMandTurb\\TESTclassification_results_LangleyRun34_filtered.npy'

# Transfer learning model for stacking on ResNet50
model1 = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
output = model1.output
output = tf.keras.layers.Flatten()(output)
resnet_model = Model(model1.input,output)

# Load the classifier
model_wp = keras.models.load_model(model_path_wp)
model_turb = keras.models.load_model(model_path_turb)

plot_flag = 1               # View the images? - slower
window_size = 3             # Moving window to filter the frames
indiv_thres = 0.85          # Individual exception threshold
confid_thres = 1.5          # SUMMED confidence over the entire window. 
                            # e.g. for 0.5 over 3 windows, make this value 1.5
use_post_process = 1        # 1 to use windowing post process, 0 if not
slice_width = 64

BL_height = 19              # boundary layer in pixels
                            # use BL_find in BL Normalization Codes

verbosity = 1               # 1 to get progress bars for each classified image, 0 if not
                            # Print-outs can help guage how fast the code is running
                            # Too many print-outs can fill up the console and result in the loss of previous print-outs
                            # "Done resizing" will still display twice for every image
                            
#%% Iterate through the list!
# Initialize some arrays for later data analysis
N_img = lines_len
WP_io_history = []
confidence_history_wp = []
wp_result_history = []
turb_io_history = []
confidence_history_turb = []
turb_result_history = []
video_statistics_WP = []
video_statistics_turb = []
OVERLAP = 0;

cut_low = 1/BL_height
cut_high = 1/(4*BL_height)

sos = signal.butter(4,[cut_high,cut_low],btype='bandpass',fs=1,output='sos')

### Iterate over all frames in the video
for i_iter in range(N_img):
    
    # Repeat essentially the same procedure of image segmentation -> feature extraction ->
    # prediction -> post-process for cohesion in space. Then, plot the frames!
    
    # Read off the image data from the labeled text file and divide into slices
    FrameList, Frame_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines, slice_width)
    
    FrameListBP = copy.deepcopy(FrameList)
    for n,slc in enumerate(FrameListBP):
        for m in range(64):
            slc[m,:] = signal.sosfiltfilt(sos,slc[m,:])
    
    # Resizes the arrays for ResNet50 compatability  
    FrameList_RS = resize_frames_for_ResNet50(FrameList, Frame_io)
    FrameListBP_RS = resize_frames_for_ResNet50(FrameListBP, Frame_io)

    # Get the features
    FrameFeaturesBP = get_bottleneck_features(resnet_model, FrameListBP_RS,  verbose=verbosity)
    FrameFeatures = get_bottleneck_features(resnet_model, FrameList_RS,  verbose=verbosity)
    
    # Calculate the confidence
    confidence_WP = model_wp.predict(FrameFeaturesBP, verbose=verbosity)
    confidence_turb = model_turb.predict(FrameFeatures, verbose=verbosity)
    
    # Apply spatial cohesion to the confidence results
    frame_res_wp, n11_wp, n10_wp, n01_wp, n00_wp = classify_WP_frame(FrameList, Frame_io, confidence_WP, window_size, indiv_thres, confid_thres, use_post_process)
    frame_res_turb, n11_turb, n10_turb, n01_turb, n00_turb = classify_WP_frame(FrameList, Frame_io, confidence_turb, window_size, indiv_thres, confid_thres, use_post_process)
    
    
    for i in range(len(frame_res_wp)):
        if frame_res_wp[i] ==1 and frame_res_turb[i] == 1:
            OVERLAP = OVERLAP+1;
            if confidence_turb[i] > confidence_WP[i]:
                frame_res_wp[i] = 0
            else:
                frame_res_turb[i] = 0          
            
    n11_wp, n10_wp, n01_wp, n00_wp= confusion_results(frame_res_wp, Frame_io)
    n11_turb, n10_turb, n01_turb, n00_turb = confusion_results(frame_res_wp, Frame_io)
    
    # Append statistics for whole-video analysis
    video_statistics_WP.append(ClassificationStatistics(n11_wp, n10_wp, n01_wp, n00_wp, len(frame_res_wp)))
    video_statistics_turb.append(ClassificationStatistics(n11_turb, n10_turb, n01_turb, n00_turb, len(frame_res_turb)))
    
    # Save off some statistics for building a ROC and PR curve later!
    confidence_history_wp.append(confidence_WP)
    confidence_history_turb.append(confidence_turb)
    wp_result_history.append(frame_res_wp)
    turb_result_history.append(frame_res_turb)
    WP_io_history.append(Frame_io)
    turb_io_history.append(Frame_io)
        
    # Plot the frame
    if plot_flag == 1: 
        # plot WP boxes and confidences
        fig, ax = plot_frame(FrameList, frame_res_wp, confidence_WP, slice_width, 1, 0)
        
        # Add on turbulence classification box rectangles and confidences
        ax.text(-57, height+112,'Turb: ', fontsize = 6)
        for i, _ in enumerate(FrameList):    
            # Add in the classification guess
            if frame_res_turb[i] == 1:
                rect = Rectangle((i*slice_width, 5), slice_width, height-10,
                                         linewidth=0.5, edgecolor='yellow', facecolor='none')
                ax.add_patch(rect)
                
            prob = round(confidence_turb[i,0],2)
            ax.text(i*slice_width+slice_width/5, height+112,f'{prob:.2f}', fontsize = 6)
    
        # Check if there's even a bounding box in the image
        if sm_bounds[0] == 'X':
            ax.set_title('Image '+str(i_iter)+'. Blue: true label. Red: NN WP. Yellow: NN Turb')
            plt.show()
            continue
        else:
            # Add the ground truth over the entire box
            ax.add_patch(Rectangle((sm_bounds[0], sm_bounds[1]), sm_bounds[2], sm_bounds[3], edgecolor='blue', facecolor='none'))
            ax.set_title('Image '+str(i_iter)+'. Blue: true label. Red: NN WP. Yellow: NN Turb')
            plt.show()
    
    
tot_frames = len(WP_io_history) * len(Frame_io);
overlap = OVERLAP/tot_frames
print('Done classifying the video!')
print(f"Percent of overlapping classification: {overlap:.6f}")

#%% Save off for breakdown work

np.save(file_save_wp, wp_result_history)
np.save(file_save_turb, turb_result_history)

#%% Save the classification results for prop speed calcs

np.save(file_save_confid, confidence_history_wp)
print('Data saved')

#%% Now make a history plot of the results using the ClassificationStatistics class

# Second Mode Stats
if np.sum(WP_io_history) == 0:
    print('Set appears to be unlabeled. Wave packet stats not run.')
else:
    N_img_WP = len(video_statistics_WP)
    acc_history_WP = [Stats.acc for Stats in video_statistics_WP]
    TP_history_WP = [Stats.TP for Stats in video_statistics_WP]
    TN_history_WP = [Stats.TN for Stats in video_statistics_WP]
    FP_history_WP = [Stats.FP for Stats in video_statistics_WP]
    FN_history_WP = [Stats.FN for Stats in video_statistics_WP]
    
    Nframe_per_img = len(FrameList)
    n = 20 # Moving avg window
        
    fig, (pl1, pl2, pl3, pl4, pl5) = plt.subplots(5,1, figsize = (16,16))
    pl1.plot(range(len(acc_history_WP)), acc_history_WP)
    pl1.plot(range(n-1, len(acc_history_WP)), moving_average(acc_history_WP, n), color='k', linewidth = 2)
    pl1.set_title('Accuracy')
    
    pl2.plot(range(len(TP_history_WP)), [img_stat/Nframe_per_img for img_stat in TP_history_WP])
    pl2.plot(range(n-1, len(acc_history_WP)), moving_average(TP_history_WP, n)/Nframe_per_img, color='k', linewidth = 2)
    pl2.set_title('True positive rate')
    
    pl3.plot(range(len(TN_history_WP)), [img_stat/Nframe_per_img for img_stat in TN_history_WP])
    pl3.plot(range(n-1, len(TN_history_WP)), moving_average(TN_history_WP, n)/Nframe_per_img, color='k', linewidth = 2)
    pl3.set_title('True negative rate')
    
    pl4.plot(range(len(FP_history_WP)), [img_stat/Nframe_per_img for img_stat in FP_history_WP])
    pl4.plot(range(n-1, len(FP_history_WP)), moving_average(FP_history_WP, n)/Nframe_per_img, color='k', linewidth = 2)
    pl4.set_title('False positive rate')
    
    pl5.plot(range(len(FN_history_WP)), [img_stat/Nframe_per_img for img_stat in FN_history_WP])
    pl5.plot(range(n-1, len(FN_history_WP)), moving_average(FN_history_WP, n)/Nframe_per_img, color='k', linewidth = 2)
    pl5.set_title('False negative rate')
    plt.show()
    
    
    # This first function grabs the FPR, TPR, and Pres, which you need to construct the ROC curves
    FPRs, TPRs, Pres = form_ROC_PR_statistics(FrameList, WP_io_history, confidence_history_wp, window_size, use_post_process)
    
    # If you just want to plot the curves, call the fcn below.
    AUC, PR, fig, ax, ax2 = form_ROC_Curve(FrameList, WP_io_history, confidence_history_wp, window_size, use_post_process)
#%%

# Turbulence Stats
if np.sum(WP_io_history) == 0:
    print('Set appears to be unlabeled. Turbulence stats not run.')
else:
    N_img_turb = len(video_statistics_turb)
    acc_history_turb = [Stats.acc for Stats in video_statistics_turb]
    TP_history_turb = [Stats.TP for Stats in video_statistics_turb]
    TN_history_turb = [Stats.TN for Stats in video_statistics_turb]
    FP_history_turb = [Stats.FP for Stats in video_statistics_turb]
    FN_history_turb = [Stats.FN for Stats in video_statistics_turb]
    
    Nframe_per_img = len(FrameList)
    n = 20 # Moving avg window
        
    fig, (pl1, pl2, pl3, pl4, pl5) = plt.subplots(5,1, figsize = (16,16))
    pl1.plot(range(len(acc_history_turb)), acc_history_turb)
    pl1.plot(range(n-1, len(acc_history_turb)), moving_average(acc_history_turb, n), color='k', linewidth = 2)
    pl1.set_title('Accuracy')
    
    pl2.plot(range(len(TP_history_turb)), [img_stat/Nframe_per_img for img_stat in TP_history_turb])
    pl2.plot(range(n-1, len(acc_history_turb)), moving_average(TP_history_turb, n)/Nframe_per_img, color='k', linewidth = 2)
    pl2.set_title('True positive rate')
    
    pl3.plot(range(len(TN_history_turb)), [img_stat/Nframe_per_img for img_stat in TN_history_turb])
    pl3.plot(range(n-1, len(TN_history_turb)), moving_average(TN_history_turb, n)/Nframe_per_img, color='k', linewidth = 2)
    pl3.set_title('True negative rate')
    
    pl4.plot(range(len(FP_history_turb)), [img_stat/Nframe_per_img for img_stat in FP_history_turb])
    pl4.plot(range(n-1, len(FP_history_turb)), moving_average(FP_history_turb, n)/Nframe_per_img, color='k', linewidth = 2)
    pl4.set_title('False positive rate')
    
    pl5.plot(range(len(FN_history_turb)), [img_stat/Nframe_per_img for img_stat in FN_history_turb])
    pl5.plot(range(n-1, len(FN_history_turb)), moving_average(FN_history_turb, n)/Nframe_per_img, color='k', linewidth = 2)
    pl5.set_title('False negative rate')
    plt.show()
    
    #the ROC and PR curves are only accurate if the video has turbulence labels
    # This first function grabs the FPR, TPR, and Pres, which you need to construct the ROC curves
    FPRs, TPRs, Pres = form_ROC_PR_statistics(FrameList, turb_io_history, confidence_history_turb, window_size, use_post_process)
    
    # If you just want to plot the curves, call the fcn below.
    AUC, PR, fig, ax, ax2 = form_ROC_Curve(FrameList, turb_io_history, confidence_history_turb, window_size, use_post_process)
    
    
