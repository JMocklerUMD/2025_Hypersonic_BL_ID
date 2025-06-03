# Version 2 of a CNN used to identify second-mode waves using ResNet50 for feature extraction

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from PIL import Image, ImageChops

import numpy as np
import os
import math
import random

#%%
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

#%% Read training data file

# Write File Name
file_name = 'C:\\Users\\cathe\\OneDrive\\Desktop\\training_data_explicit.txt'
if os.path.exists(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len = len(lines)
print(f"{lines_len} lines read")


#%% Write training data to required arrays and visualize

WP_io = []
#SM_bounds_Array = []
Imagelist = []

for i in range(200):
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

                        

#%%
omit_array = []
for i in range(len(Imagelist)):
    if Imagelist[i].shape != (64, 64):
        omit_array.append(i)

Imagelist = [element for i, element in enumerate(Imagelist) if i not in omit_array]
WP_io = [element for i, element in enumerate(WP_io) if i not in omit_array]


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
Imagelist,WP_io = Shuffler(Imagelist, WP_io)
Imagelist = np.array(Imagelist)
WP_io = np.array(WP_io)
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
# Implementing an early stopping monitor (optional for now)
early_stopping_monitor = EarlyStopping(patience = 3)

#%%
"""
Let's train the model and take a look at how it does
"""

# Running the functions to bring in our images and labels
trainimgs = Imagelist_resized
trainlbls = WP_io

trainimgs, testimgs, trainlbs, testlbls = train_test_split(Imagelist_resized,WP_io, test_size=0.2, random_state=69)

trainimgs_res = get_bottleneck_features(resnet_model, trainimgs)

#%%

# Number of Epochs to Train on:
ne = 30

#%%

# Training the classification model and checking accuracy
history = model.fit(trainimgs_res, trainlbs, validation_split = 0.2, epochs = ne, verbose = 1)

#%%
# Generating a range of epochs run
epoch_list = list(range(1,ne + 1))

model.save('ClassifierV1m.h5')

#%%
# Making some plots to show our results
f, (pl1, pl2) = plt.subplots(1, 2, figsize = (15,4), gridspec_kw = {'wspace': 0.3})
t = f.suptitle('Neural Network Performance', fontsize = 14)
# Accuracy Plot
pl1.plot(epoch_list, history.history['accuracy'], label = 'train accuracy')
pl1.plot(epoch_list, history.history['val_accuracy'], label = 'validation accuracy')
pl1.set_xticks(np.arange(0, ne + 1, 5))
pl1.set_xlabel('Epoch')
pl1.set_ylabel('Accuracy')
pl1.set_title('Accuracy')
leg1 = pl1.legend(loc = "best")
# Loss plot for classification
pl2.plot(epoch_list, history.history['loss'], label = 'train loss')
pl2.plot(epoch_list, history.history['val_loss'], label = 'validation loss')
pl2.set_xticks(np.arange(0, ne + 1, 5)) 
pl2.set_xlabel('Epoch')
pl2.set_ylabel('Loss')
pl2.set_title('Classification Loss')
leg2 = pl2.legend(loc = "best")
plt.show()