# MODEL DEVELOPMENT CLASSIFIER

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import os
import random
import time

import keras

import tensorflow as tf

from keras import optimizers, layers, regularizers
from keras.applications import resnet50 
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Model
from keras.models import Sequential

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import copy

# Update to project folder location
base_folder = 'C:\\Users\\tyler\\Desktop\\NSSSIP25\\Machine Learning Classification - NSSSIP25'

os.chdir(base_folder)

# Import custom functions
from ML_utils import *
from Results_utils import *
from ML_models import *


#%% Initialize basic training information and file paths, indicate which model is being trained
second_mode = True
turb = False

# file location where the finished model should be saved to
model_path = f'{base_folder}\\Example Data and Outputs\\Build_Classifier\\TESTsecondmodemodel.keras'
# file of the training data
training_data = f'{base_folder}\\Example Data and Outputs\\LangleyRun34_filtered_training_data.txt'
N_img = 200     # how many images to train on
if turb:
    print('Finding turbulence')
if second_mode:
    print('Finding second-mode waves')
    
# Training parameters
slice_width = 64        # Width of slice that's classified
ne = 20                 # Number of epochs to train


# Set processing parameters for whole-set classification
plot_flag = 1               # View the images? MUCH SLOWER

use_post_process = 1        # 1 to use windowing post process, 0 if not
window_size = 3             # Moving window to filter the frames
indiv_thres = 0.85          # Individual exception threshold
confid_thres = 1.5          # SUMMED confidence over the entire window. 
                            # e.g. for 0.5 over 3 windows, make this value 1.5
                            
verbosity = 1               # 1 to get two progress bars for each classified image, 0 if not
                            # Print-outs can help guage how fast the code is running
                            # Too many print-outs can fill up the console and result in the loss of previous print-outs
                            # "Done resizing" will still display for every image

#%% READ IN THE DATA SET AND SAMPLE THE IMAGES
start_time = time.time()

# Read in the second mode wave packet labeled file
lines, lines_len = write_text_to_code(training_data)

# Initialize some arrays that we will sample from
io, ImageList, sampled_list = [], [], []
N_tot = lines_len
i_sample, img_count = 0, 0
N_img = N_img

# Sample from the distribution: break when we reach N_imgs sampled or when we 
# Run out of frames to sample from!
while (img_count < N_img) and (i_sample < N_tot):
    
    # Randomly sample image with probability N_img/N_tot
    # Skip image if in the 1-N_img/N_tot probability
    if np.random.random_sample(1)[0] < (1-N_img/N_tot):
        i_sample = i_sample + 1
        continue
    
    # Read off the image data from the labeled text file and divide into slices
    FrameList, Frame_io, slice_width, height, bounds = image_splitting(i_sample, lines, slice_width)
    
    # Now append to the sampled list
    for j in range(len(FrameList)):
        io.append(Frame_io[j])
        ImageList.append(FrameList[j])
    
    # Increment to the next sample image and image count
    i_sample = i_sample + 1
    img_count = img_count + 1
    
    # Keep track of which images were sampled from
    sampled_list.append(i_sample)
    
# Completed! Return some basic stats on our train/test data set
print(f'N_img = {N_img}')
print(f'Number of used images: {img_count}')
print(f'slice_width  = {slice_width}')
print('Done sampling images! Now resize for ResNet50 compatability')

# Resizes the arrays for ResNet50 compatability  
ImageList_Resized = resize_frames_for_ResNet50(ImageList, io)

# Now split the images
trainimgs, testimgs, trainlbs, testlbls = train_test_split(ImageList_Resized, io, test_size=0.2, random_state=70)

print("Finished splitting into a train/test set")

#%% TRAIN THE CLASSIFICATION MODEL

# First, we need to extract the top-layer features to train our NN. 
# We use resnet50 for this!
model1 = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
output = model1.output
output = tf.keras.layers.Flatten()(output)
resnet_model = Model(model1.input,output)

# Locking in the weights of the feature detection layers
resnet_model.trainable = False
for layer in resnet_model.layers:
	layer.trainable = False

# Now we extract the features from our image set
trainimgs_resnet50_features = get_bottleneck_features(resnet_model, trainimgs, verbose=1)
testimgs_resnet50_features  = get_bottleneck_features(resnet_model, testimgs,  verbose=1)

# Now we train the model. All model training and parameters are stored in ML_models.py
# Only the number of epochs, ne, is passed in.
history, model, ne = feature_extractor_training(resnet_model, trainimgs_resnet50_features, trainlbs, ne)

if second_mode:
    print("Second-mode Wave Model Training Complete!")

if turb:
    print("Turbulence Model Training Complete!")

end_time = time.time()
print(f'Time to read in data, process, and train model: {end_time-start_time} seconds')

#%% FIRST VALIDATION: CLASSIFY THE TEST SET

# First plot to see how well the training did
if second_mode:
    plot_training_history(history, ne, 'Second-mode')
if turb:
    plot_training_history(history, ne, 'Turbulence')

# Now let's pass the test image resnet features through to see how well it performed
test_res = model.predict(testimgs_resnet50_features)
test_res_binary = np.round(test_res)

# Compute the basic binary classification results
n11, n10, n01, n00 = confusion_results(test_res_binary, testlbls)

# We save off the classification performance statistics in an object
class_res = ClassificationStatistics(n11, n10, n01, n00, len(testlbls))
class_res.print_stats()
# All of the printed values are accessable attributes in this class!

#%% SECOND VALIDATION: CLASSIFY ALL LABELED IMAGES
# This script demonstrates how to classify an entire labeled set

# Read in the entire file name
lines, lines_len = write_text_to_code(training_data)

N_frames = lines_len        # Number of frames to go through for whole-set


print(f'Classifying {N_frames} images (this may take a while)...')
start_time = time.time()

video_statistics = []
io_history = []
confidence_history = []
classification_history = []

### Iterate over all frames in the video
for i_iter in range(N_frames):
    
    # Repeat essentially the same procedure of image segmentation -> feature extraction ->
    # prediction -> post-process for cohesion in space. Then, plot the frames!
    
    # Read off the image data from the labeled text file and divide into slices
    FrameList, Frame_io, slice_width, height, bounds = image_splitting(i_iter, lines, slice_width)
    
    # Resizes the arrays for ResNet50 compatability  
    FrameList_RS = resize_frames_for_ResNet50(FrameList, io)
    
    # Get the features
    FrameFeatures = get_bottleneck_features(resnet_model, FrameList_RS,  verbose=verbosity)
    
    # Calculate the confidence
    confidence = model.predict(FrameFeatures, verbose=verbosity)
    
    # Apply spatial cohesion to the confidence results
    frame_res, n11, n10, n01, n00 = classify_WP_frame(FrameList, Frame_io, confidence, window_size, indiv_thres, confid_thres, use_post_process)
    # Append statistics for whole-video analysis
    video_statistics.append(ClassificationStatistics(n11, n10, n01, n00, len(frame_res)))
    
    # Save off some statistics for building a ROC and PR curve later!
    confidence_history.append(confidence)
    io_history.append(Frame_io)
    classification_history.append(frame_res)
    
    # Plot the frame
    if plot_flag == 1:
        fig, ax = plot_frame(FrameList, frame_res, confidence, slice_width, second_mode, turb)
    
        # Check if there's even a bounding box in the image
        if bounds[0] == 'X':
            if second_mode and turb:
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN WP. Orange: NN Turbulence')
            elif second_mode:
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN WP')
            else:
                ax.set_title('Image '+str(i_iter)+'. Blue: Labeled Turbulence')
            plt.show()
            continue
        else:
            # Add the ground truth over the entire box
            ax.add_patch(Rectangle((bounds[0], bounds[1]), bounds[2], bounds[3], edgecolor='blue', facecolor='none'))
            if second_mode and turb:
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN WP. Orange: NN Turbulence')
            elif second_mode:
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN WP')
            else:
                ax.set_title('Image '+str(i_iter)+'. Blue: Labeled Turbulence')
            plt.show()
            

#%% Now make a history plot of the results using the ClassificationStatistics class
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

#%% Construct the ROC Curve

# This first function grabs the FPR, TPR, and Pres, which you need to construct the ROC curves
FPRs, TPRs, Pres = form_ROC_PR_statistics(FrameList, io_history, confidence_history, window_size, use_post_process)

# If you just want to plot the curves, call the fcn below.
AUC, PR, fig, ax, ax2 = form_ROC_Curve(FrameList, io_history, confidence_history, window_size, use_post_process)

#%%

#save model 
model.save(model_path)
print('Model saved')
  

# #%% Save off a series of frames for the poster
# # Choose the start and end locs

# # start_vid, stop_vid = 95, 104
# # for i_iter in range(start_vid, stop_vid):
# #     # Split the image and classify the slices
# #     Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines, slice_width)

# #     imageReconstruct = np.hstack([image for image in Imagelist])
    
# #     fig, ax = plt.subplots(1, figsize=(16, 4))
# #     ax.imshow(imageReconstruct, cmap = 'gray')
    
# #     classification_result = filtered_result_history[i_iter]
    
# #     # Add on classification box rectangles
# #     for i, _ in enumerate(Imagelist):    
# #         # Add in the classification guess
# #         if classification_result[i] == 1:
# #             rect = Rectangle((i*slice_width, 1), slice_width, height-3,
# #                                      linewidth=0.5, edgecolor='red', facecolor='none')
# #             ax.add_patch(rect)
   
# #     # Add the ground truth
# #     ax.add_patch(Rectangle((sm_bounds[0],sm_bounds[1]), sm_bounds[2], sm_bounds[3], edgecolor='blue', facecolor='none'))
# #     ax.tick_params(axis='both', labelsize=8) # Change 8 to your desired smaller size
    
# #     # CHANGE THIS TO YOUR FILE PATH WHERE YOU WANT TO SAVE THE IMAGES
# #     plt.savefig("C:\\Users\\Desktop\\file\\plotted_img"+str(i_iter)+".svg")