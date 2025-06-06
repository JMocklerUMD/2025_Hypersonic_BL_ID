# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 14:46:41 2025

@author: cathe
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
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
#%%

file_name = 'C:\\Users\\cathe\\OneDrive\\Desktop\\training_data_explicit.txt'
if os.path.exists(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len = len(lines)
print(f"{lines_len} lines read")


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
    Imagelist.append(image_line)
    
    if sm_check.startswith('X'):
        WP_io.append(0)
        plt.show()
        print('No wave-packet detected')
    else:
        WP_io.append(1)
        SM_bounds_Array.append(sm_bounds)
        rect = plt.Rectangle((sm_bounds[0], sm_bounds[1]), sm_bounds[2], sm_bounds[3], edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        fig.canvas.draw()
        plt.show()

#%%
omit_array = []
for i in range(len(Imagelist)):
    if Imagelist[i].shape != (64, 1280):
        omit_array.append(i)

Imagelist = [element for i, element in enumerate(Imagelist) if i not in omit_array]
WP_io = [element for i, element in enumerate(WP_io) if i not in omit_array]
SM_bounds_Array = [element for i, element in enumerate(SM_bounds_Array) if i not in omit_array]


#%%
import tensorflow as tf

from tensorflow.keras import optimizers

from tensorflow.keras.applications import resnet50, vgg16

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, Input

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
#import cv2

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
	print('Getting Feature Data From CNN...')
	features = model.predict(input_imgs, verbose = 1)
	return features

def img_preprocess(input_image):
    input_image = np.stack((input_image,)*3,axis = -1)
    input_image = array_to_img(input_image)
    #input_image = input_image.resize((224,224))
    input_image = img_to_array(input_image)
    input_image = input_image / 255
    return input_image
#%%

Imagelist,WP_io,SM_bounds_Array = Shuffler(Imagelist, WP_io, SM_bounds_Array)
Imagelist = np.array(Imagelist)
WP_io = np.array(WP_io)
SM_bounds_Array = np.array(SM_bounds_Array)
Imagelist_resized = np.array([img_preprocess(img) for img in Imagelist])

#%%

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

trainimgs = Imagelist_resized
trainlbls = WP_io
trainimgs, testimgs, trainlbs, testlbls = train_test_split(Imagelist_resized, WP_io, test_size=0.25, random_state=69)
ne = 30

#input layer
input_layer = Input(shape=(64,1280,3))

#convolution
x = Conv2D(32, (3,3), activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(128, (3,3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Flatten()(x)
output = Dense(1, activation = 'sigmoid')(x)

cnn_model = Model(inputs = input_layer, outputs=output)
cnn_model.compile(optimizer = 'adam',
                  loss = 'binary_crossentropy', # only looking for 1s and 0s rn
                  metrics = ['accuracy'])
#%%
early_stopping_monitor = EarlyStopping(patience = 3)

history = cnn_model.fit(trainimgs, 
                        trainlbls,
                        validation_split = 0.2, 
                        epochs = 30,
                        verbose = 1)

#%% 
# Plotting accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


