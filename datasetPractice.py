#%% Import libraries
import numpy as np
import os
import math
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import tensorflow as tf

import keras
from keras import optimizers, layers, regularizers
from keras.applications import ResNet50, resnet50
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

print('Done importing libraries')

#%% Define functions
def write_data(file_name, N_img, slice_width):
    '''
    Pass in file_name and N_img
    Reads and splits to arrays
    Splits to train and test
    Outputs train images, train labels, test images, test labels, and lines_len
    '''
    
    # Read training data file
    # PREPROCESSING: the following block of codes accept the image data from a 
    # big text file, parse them out, then process them into an array
    # that can be passed to the keras NN trainer

    print('Reading training data file')

    # Write File Name
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()
    else:
        raise ValueError("No training_data file detected")

    lines_len = len(lines)
    print(f"{lines_len} lines read")
    
    print('Begin writing training data to numpy array')
    
    WP_io = []
    #SM_bounds_Array = []
    Imagelist = []
    N_tot = lines_len
    i_sample, img_count = 0, 0
    sampled_list = []
    
    # Break when we aquire 100 images or when we run thru the 1000 frames
    while (img_count < N_img) and (i_sample < N_tot):
        
        # Randomly sample image with probability N_img/N_tot
        # Skip image if in the 1-N_img/N_tot probability
        if np.random.random_sample(1)[0] < (1-N_img/N_tot):
            i_sample = i_sample + 1
            continue
        
        # Otherwise, we accept the image and continue with the processing
        curr_line = i_sample;
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
        #    print(f"Skipping image at line {i_sample+1} — unexpected size {full_image.shape}")
        #    continue
        
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
                if x_max >= x_start+slice_width/4 and x_min <= x_end-slice_width/4:
                    WP_io.append(1)
    
                else:
                    WP_io.append(0)
        
        # Increment to the next sample image and image count
        i_sample = i_sample + 1
        img_count = img_count + 1
        
        # Inspect what images were selected later
        sampled_list.append(i_sample)
    print(f'N_img = {N_img}')
    print(f'Number of used images: {img_count}')
    print(f'slice_width  = {slice_width}')
    
    print('Done sampling images!')
    
    # Resizes the arrays
    Imagelist = np.array(Imagelist)
    WP_io = np.array(WP_io)
    Imagelist_resized = np.array([img_preprocess(img) for img in Imagelist])
    print("Done Resizing")
    
    return Imagelist_resized, WP_io, lines_len, img_count

def img_preprocess(input_image):
    '''
    Reshapes the images for ResNet50 input
    INPUTS:  input_image:       (height,slice_width) numpy array of double image data prescaled between [0,1]     
    OUTPUTS: processed_image:   (224, 224, 3) numpy array of the stretched input data
    '''
    input_image = np.stack((input_image,)*3,axis = -1)
    input_image = array_to_img(input_image)
    input_image = input_image.resize((224,224))
    processed_image = img_to_array(input_image)
    #input_image = (input_image / 127.5) - 1
    return processed_image

def augment_img(image, label):
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label
#%% Input files and run settings

file_name = "C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\wavepacket_labels_combined.txt"
N_imgs = 50
slice_width = 128

Imagelist_resized, WP_io, lines_len, used_img_count = write_data(file_name, N_imgs, slice_width)
images = Imagelist_resized
labels = WP_io

#%% add to dataset

dataset = tf.data.Dataset.from_tensor_slices((images,labels))

BATCH_SIZE = 32

dataset = dataset.map(augment_img, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)


#%% define model

def build_resnet50_binary_classifier(input_shape=(224, 224, 3), freeze_base=True):
    # Base ResNet50 (no top, pretrained on ImageNet)
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    
    if freeze_base:
        base_model.trainable = False  # freeze all layers

    # Custom classification head
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)  # Important: training=False keeps batchnorm frozen
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification

    model = Model(inputs, outputs)
    return model

model = build_resnet50_binary_classifier(input_shape=(224, 224, 3))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


#%% train model
model.fit(dataset, epochs=10)
