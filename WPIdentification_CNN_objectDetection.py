# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:12:14 2024

@author: Ryan de Silva
"""

# Version 2 of a CNN used to identify second-mode waves using ResNet50 for feature extraction

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

import keras
from keras import regularizers

from skimage.transform import resize

from PIL import Image, ImageChops

import numpy as np
import os
import math
import random

#%%

def Shuffler(list1, list2, list3):
	n1 = list1
	n2 = list2
	n3 = list3
	a = []
	for i in range(0,len(n1)):
		temp = [n1[i],n2[i],n3[i]]
		a.append(temp)
	random.shuffle(a)
	n1new = []
	n2new = []
	n3new = []
	for i in range(0,len(a)):
		n1new.append(a[i][0])
		n2new.append(a[i][1])
		n3new.append(a[i][2])

	return n1new, n2new, n3new

#%% Read training data file

# Write File Name
file_name = "C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\Test1\\run33\\turbulence_training_data_only_positives.txt"
if os.path.exists(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len = len(lines)
print(f"{lines_len} lines read")

#%% Write training data to required arrays

WP_io = []
SM_bounds_Array = []
Imagelist = []

for i in range(lines_len):
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
    image_line = np.array(image_data).astype(np.float64)
    image_line = image_line.reshape(image_size)  # Reshape to (rows, columns)

    print(f"Number of Parts: {len(parts)}")
    print(f"Run: {run}")
    print(f"SM Bounds: {sm_bounds}")
    print(f"Image Size: {image_size}")

    '''
    vmin = np.min(image_line)
    vmax = np.max(image_line)
    # Figure specs
    stretch_y = 2.5;
    aspect_ratio = stretch_y*image_size[0]/image_size[1]
    figsize = (20,20*aspect_ratio)
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    # Display image
    ax.imshow(image_line, cmap='bone', vmin=vmin, vmax=vmax)
    '''
    Imagelist.append(image_line)
    
    
    if sm_check.startswith('X'):
        WP_io.append(0)
        #plt.show()
        print('No wave-packet detected')
    else:
        WP_io.append(1)
        SM_bounds_Array.append(sm_bounds)
        '''
        rect = plt.Rectangle((sm_bounds[0], sm_bounds[1]), sm_bounds[2], sm_bounds[3], edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        fig.canvas.draw()
        plt.show()
        '''

#%%
'''
omit_array = []
for i in range(len(Imagelist)):
    if Imagelist[i].shape != (64, 1200):
        omit_array.append(i)

Imagelist = [element for i, element in enumerate(Imagelist) if i not in omit_array]
WP_io = [element for i, element in enumerate(WP_io) if i not in omit_array]
SM_bounds_Array = [element for i, element in enumerate(SM_bounds_Array) if i not in omit_array]
'''

#%%
import tensorflow as tf

from tensorflow.keras import optimizers

from tensorflow.keras.applications import resnet50, vgg16

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img

import matplotlib.pyplot as plt

import h5py

import numpy as np

from sklearn.model_selection import train_test_split
import cv2

#%%

"""
Progress bar function for use in future functions
"""

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
    input_image = input_image / 255
    return input_image


#%%
Imagelist,WP_io,SM_bounds_Array = Shuffler(Imagelist, WP_io, SM_bounds_Array)
Imagelist = np.array(Imagelist)
WP_io = np.array(WP_io)
SM_bounds_Array = np.array(SM_bounds_Array)

#%% Code to just take the last 224 pixels of each image and resize bbox

# take last 224 pixels
Imagelist_temp = []
for i, img in enumerate(Imagelist):
    img = img[0:64, 976:1200]
    Imagelist_temp.append(img)
Imagelist = Imagelist_temp
    
delete_array = []
SM_bounds_Array = np.array(SM_bounds_Array).astype(np.float64)
for i, lbl in enumerate(SM_bounds_Array):
        dif = 976 - lbl[0]
        lbl[0] = lbl[0]-976
        if lbl[0] < 0:
            lbl[0] = 0
        lbl[2] = lbl[2] - dif #resize width accordingly
        if lbl[2] < 0:    # check to make sure width is reasonable - if not turb is not visible in cropped image so remove
            delete_array.append(i)
        
        #rescale to [0,1] 
        lbl[0] = lbl[0]/224.0
        lbl[2] = lbl[2]/224.0
        lbl[1] = lbl[1]/64.0
        lbl[3] = lbl[3]/64.0
            
for _,i in enumerate(reversed(delete_array)): #delete backwards so as to not mess up indexing
    Imagelist = np.delete(Imagelist, i, axis=0)
    WP_io = np.delete(WP_io,i,axis=0)
    SM_bounds_Array = np.delete(SM_bounds_Array, i, axis=0)

#%%
Imagelist_resized = np.array([img_preprocess(img) for img in Imagelist])

#%%
"""
Building the model below
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

#%%
# Generate an input shape for our classification layers
input_shape = resnet_model.output_shape[1]

# Now we'll add new classification layers
model = Sequential()
model.add(InputLayer(input_shape = (input_shape,)))
model.add(Dense(256, activation = 'relu', input_dim = input_shape))
model.add(Dropout(0.3))
model.add(Dense(128, activation = 'relu', input_dim = input_shape))
model.add(Dropout(0.3))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))

# Compiling our masterpiece
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

# model.summary()
#%%
# Second set of layers for the bounding box coordinates
reg_weight = 1e-4

model2 = Sequential()
model2.add(InputLayer(input_shape = (input_shape,)))
model2.add(Dense(256, activation = 'relu',kernel_regularizer=regularizers.L1L2(l1=reg_weight, l2=reg_weight),     # Regularization penality term
                   bias_regularizer=regularizers.L2(reg_weight)))
model2.add(Dropout(0.5))
model2.add(Dense(256, activation = 'relu',kernel_regularizer=regularizers.L1L2(l1=reg_weight, l2=reg_weight),     # Regularization penality term
                   bias_regularizer=regularizers.L2(reg_weight)))
model2.add(Dropout(0.5))
model2.add(Dense(4, activation = 'sigmoid'))

huber_loss = tf.keras.losses.Huber(delta=1.0)
#mse

# Compiling our second masterpiece
# RMSprop
model2.compile(optimizer = optimizers.Adam(learning_rate = 1e-3), loss = huber_loss)

#%%
# Implementing an early stopping monitor (optional for now)
early_stopping_monitor = EarlyStopping(patience = 3)


#%%
"""
Let's train the model and take a look at how it does
"""

# Running the functions to bring in our images and labels
trainimgs = Imagelist_resized
trainlbs = WP_io
boxcoords = SM_bounds_Array

trainimgs, testimgs, trainlbs, testlbs, boxcoords_train, boxcoords_test = train_test_split(Imagelist_resized,WP_io,SM_bounds_Array, test_size=0.25, random_state=69)

trainimgs_res = get_bottleneck_features(resnet_model, trainimgs)

#%%

# Number of Epochs to Train on:
ne = 15

#%%

# Training the classification model and checking accuracy
history = model.fit(trainimgs_res, trainlbs, epochs = ne, verbose = 1)

#%%

# Doing the same with the bounding box model
history2 = model2.fit(trainimgs_res, boxcoords_train, validation_split = 0.2, epochs = ne, verbose = 1)

#%%
# Generating a range of epochs run
epoch_list = list(range(1,ne + 1))

#model.save('ClassifierV1m.h5')
#model2.save('RegressorV1m.h5')

#%%
# Making some plots to show our results
f, (pl1, pl2, pl3) = plt.subplots(1, 3, figsize = (15,4), gridspec_kw = {'wspace': 0.3})
t = f.suptitle('Neural Network Performance', fontsize = 14)
# Accuracy Plot
pl1.plot(epoch_list, history.history['accuracy'], label = 'train accuracy')
#pl1.plot(epoch_list, history.history['val_accuracy'], label = 'validation accuracy')
pl1.set_xticks(np.arange(0, ne + 1, 5))
pl1.set_xlabel('Epoch')
pl1.set_ylabel('Accuracy')
pl1.set_title('Accuracy')
leg1 = pl1.legend(loc = "best")
# Loss plot for classification
pl2.plot(epoch_list, history.history['loss'], label = 'train loss')
#pl2.plot(epoch_list, history.history['val_loss'], label = 'validation loss')
pl2.set_xticks(np.arange(0, ne + 1, 5)) 
pl2.set_xlabel('Epoch')
pl2.set_ylabel('Loss')
pl2.set_title('Classification Loss')
leg2 = pl2.legend(loc = "best")
# Loss plot for bounding boxes
pl3.plot(epoch_list, history2.history['loss'], label = 'train loss')
pl3.plot(epoch_list, history2.history['val_loss'], label = 'validation loss')
pl3.set_xticks(np.arange(0, ne + 1, 5))
pl3.set_xlabel('Epoch')
pl3.set_ylabel('Loss')
pl3.set_title('Regression Loss')
leg3 = pl3.legend(loc = "best")
plt.show()

# Displaying a sample image with boxes drawn on it
# demoimg = cv2.imread('Processed_4017\\Processed_SMW_Present_2.tif')
# demobox = boxcoords[1]
# cv2.rectangle(demoimg,(demobox[0],demobox[1]),(demobox[2]+demobox[0], demobox[3]+demobox[1]),(0,255,0),2)
# cv2.rectangle(demoimg,(demobox[0],demobox[1]),(demobox[2]+demobox[0], demobox[3]+demobox[1]),(255,0,0),2)
# cv2.imshow('test',demoimg)
# cv2.waitKey()


# Passing new images to the network for predictions
#predimgs, predlbls, inputboxcoords = read_image_data('Processed_Predictions','Processed_Predictions_Bounding_Boxes.txt')
trainimgs_res = get_bottleneck_features(resnet_model, trainimgs)
score = model.evaluate(trainimgs_res, trainlbs, verbose = 1)
print(score[1]*100, '\n')
predboxcoords = model2.predict(trainimgs_res)
for i in range(0,len(trainlbs)):
    inputbox = boxcoords_train[i]
    outputbox = predboxcoords[i].astype(int)
    
    #resize bounding boxes
    inputbox[0] = inputbox[0]*224.0
    inputbox[2] = inputbox[2]*224.0
    outputbox[0] = outputbox[0]*224.0
    outputbox[2] = outputbox[2]*224.0
    
    inputbox[1] = inputbox[1]*64.0
    inputbox[3] = inputbox[3]*64.0
    outputbox[1] = outputbox[1]*64.0
    outputbox[3] = outputbox[3]*64.0
    
    print(f'Input {i}: {inputbox}')
    print(f'Output {i}: {outputbox}')
    
    image = trainimgs[i]
    #image = resize(image, (64, 224), anti_aliasing=True)
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    fig.suptitle('Object detection - Blue: labeled, Red: NN Turbulence')
    rect = patches.Rectangle((inputbox[0],inputbox[1]), inputbox[2], inputbox[3], linewidth=2, edgecolor='blue', facecolor='none')    
    ax.add_patch(rect) 
    rect = patches.Rectangle((outputbox[0],outputbox[1]), outputbox[2], outputbox[3], linewidth=2, edgecolor='red', facecolor='none')    
    ax.add_patch(rect) 
    plt.suptitle(f'Image {i}')
    plt.show()
    
    '''
    cv2.rectangle(image,(inputbox[0],inputbox[1]),(inputbox[2]+inputbox[0], inputbox[3]+inputbox[1]),(0,255,0),2)
    cv2.rectangle(image,(outputbox[0],outputbox[1]),(outputbox[2]+outputbox[0], outputbox[3]+outputbox[1]),(255,0,0),2)
    cv2.imshow('test', image)
    cv2.waitKey()'''
