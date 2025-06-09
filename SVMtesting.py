# Version 2 of a CNN used to identify second-mode waves using ResNet50 for feature extraction
#trying to use an SVM


import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from PIL import Image, ImageChops

import numpy as np
import os
import math
import random

import tensorflow as tf

from keras import optimizers

from keras.applications import resnet50, vgg16

from keras.callbacks import EarlyStopping

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer

from keras.models import Model
from keras.models import Sequential

from keras.preprocessing import image

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

import matplotlib.pyplot as plt

import h5py

import numpy as np

from sklearn.model_selection import train_test_split
#import cv2


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

#%% Function calls
'''
Function calls used throughout the script.
'''
def Shuffler(list1, list2):
	n1 = list1
	n2 = list2
	a = []
	for i in range(0,len(n1)):
		temp = [n1[i],n2[i]]
		a.append(temp)
	random.shuffle(a)
	n1new = []
	n2new = []
	for i in range(0,len(a)):
		n1new.append(a[i][0])
		n2new.append(a[i][1])

	return n1new, n2new

def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '>', printEnd = "\r"):
	percent = ("{0:." + str(decimals) + "f}").format(100*(iteration/float(total)))
	filledLength = int(length*iteration//total)
	bar = fill*filledLength + '-'*(length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
	if iteration == total:
		print()

# This function tells our feature extractor to do its thing
def get_bottleneck_features(model, input_imgs):
	print('Getting Feature Data From ResNet...')
	features = model.predict(input_imgs, verbose = 1)
	return features

def img_preprocess(input_image):
    input_image = np.stack((input_image,)*3,axis = -1)
    input_image = array_to_img(input_image)
    input_image = input_image.resize((224,224))
    input_image = img_to_array(input_image)
    #input_image = (input_image / 127.5) - 1
    return input_image


#%% Read training data file
'''
Preprocessing: the following block of codes accept the image data from a 
big text file, parse them out, then process them into an array
that can be passed to the keras NN trainer
'''

print('Reading training data file')

# Write File Name
file_name = r"C:\Users\tyler\Desktop\NSSSIP25\Training Data ML Wave-Packet Identification-selected\training_data_explicit.txt"
if os.path.exists(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len = len(lines)
print(f"{lines_len} lines read")


#%% Write training data to required arrays and visualize
print('Begin writing training data to numpy array')

WP_io = []
#SM_bounds_Array = []
Imagelist = []
N_img = 125

for i in range(N_img):
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
    
    if full_image.shape != (64, 1280):
        print(f"Skipping image at line {i+1} — unexpected size {full_image.shape}")
        continue
    
    slice_width = 64
    height, width = full_image.shape
    num_slices = width // slice_width
    
    # Only convert bounds to int if not sm_check.startswith('X')
    if not sm_check.startswith('X'):
        sm_bounds = list(map(int, sm_bounds))
        x_min, y_min, box_width, box_height = sm_bounds
        x_max = x_min + box_width
        y_max = y_min + box_height
    
    for i in range(num_slices):
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
            if x_max >= x_start and x_min <= x_end:
                WP_io.append(1)

            else:
                WP_io.append(0)
                
    # Track progress
    if (i % 10) == 0:
        print(f"{i/N_img} percent complete")


#%% Catches any arrays that are not correct size
omit_array = []
for i in range(len(Imagelist)):
    if Imagelist[i].shape != (64, 64):
        omit_array.append(i)

Imagelist = [element for i, element in enumerate(Imagelist) if i not in omit_array]
WP_io = [element for i, element in enumerate(WP_io) if i not in omit_array]


#%% Resizes the arrays
# Imagelist,WP_io = Shuffler(Imagelist, WP_io)
# Keras should shuffle our images for us - probably don't need to do!
Imagelist = np.array(Imagelist)
WP_io = np.array(WP_io)
Imagelist_resized = np.array([img_preprocess(img) for img in Imagelist])

#%%
"""
Building the Resnet50 model: images are first passed through the Reset50 model
prior to passing through one last NN layer that we will define. Initialize
this code block once!
"""

# Bringing in ResNet50 to use as our feature extractor
model1 = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
output = model1.output
output = tf.keras.layers.Flatten()(output)
resnet_model = Model(model1.input,output)

# Locking in the weights of the feature detection layers
resnet_model.trainable = False
for layer in resnet_model.layers:
	layer.trainable = False
    
# Split the images - do this once to avoid memory allocation issues!
# Running the functions to bring in our images and labels
trainimgs = Imagelist_resized
trainlbls = WP_io

trainimgs, testimgs, trainlbs, testlbs = train_test_split(Imagelist_resized,WP_io, test_size=0.2, random_state=69)

trainimgs_res = get_bottleneck_features(resnet_model, trainimgs)
testimgs_res = get_bottleneck_features(resnet_model, testimgs)

#%%
'''
Train SVM - below code is modified from ChatGPT 
'''
# Standardize the features
print("Standarizing features...")
scaler = StandardScaler()
trainimgs_scaled = scaler.fit_transform(trainimgs_res)
testimgs_scaled = scaler.transform(testimgs_res)

# Train SVM
svm = SVC(kernel='linear')
print("Fitting SVM (this might take a while)...")
svm.fit(trainimgs_scaled, trainlbs)

#%%
print('Running statistics...')
y_pred = svm.predict(testimgs_scaled)
acc = accuracy_score(testlbs, y_pred)
print(f"SVM Accuracy: {acc:.4f}")

#%%
#more stats - PROBABLY NEEDS TO BE DOUBLE CHECKED
TN,FP,FN,TP=0,0,0,0
for i in enumerate(testlbs):
    if testlbs[i] == 0:
        if y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 1:
            FP += 1 
    elif testlbs[i] == 1:
        if y_pred[i] == 0:
            FN += 1
        if y_pred[i] == 1:
            TP += 1
          
true_positive = TP/len(testlbs)
true_negative = TN/len(testlbs)

print("True positive rate: {true_positive} - this may need to be corrected")
print("True negative rate: {true_negative}")

#%%
#save trained SVM
import joblib
joblib.dump(svm, r"C:\Users\tyler\Desktop\NSSSIP25\SVM1000.joblib")
#svm = joblib.load('file_path')
