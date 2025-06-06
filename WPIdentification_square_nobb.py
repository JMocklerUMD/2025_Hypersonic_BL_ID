# Version 2 of a CNN used to identify second-mode waves using ResNet50 for feature extraction

#CHANGE THIS VAlUES (see also lines ~22 and ~205)
pixelSize = 224


import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from PIL import Image, ImageChops

import numpy as np
import os
import math
import random

import tensorflow as tf

from keras import optimizers

#CHANGE THIS VALUE
from keras.applications import ResNet50

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
	print('Getting Feature Data From Model...')
	features = model.predict(input_imgs, verbose = 1)
	return features

def img_preprocess(input_image):
    input_image = np.stack((input_image,)*3,axis = -1)
    input_image = array_to_img(input_image)
    input_image = input_image.resize((pixelSize,pixelSize))
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
file_name = 'C:\\Users\\tyler\\Desktop\\NSSSIP25\\Training Data ML Wave-Packet Identification-selected\\training_data_explicit.txt'
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
    
    slice_width = 128
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
    if Imagelist[i].shape != (64, slice_width):
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

# Bringing in model to use as our feature extractor
# CHANGE HERE
model1 = ResNet50(include_top = False, weights ='imagenet', input_shape = (pixelSize,pixelSize,3))
output = model1.output
output = tf.keras.layers.Flatten()(output)
model = Model(model1.input,output)

#load pretrained inat2021_mini_supervised_from_scratch weights from https://github.com/visipedia/newt/blob/main/benchmark/README.md
#model.load_weights('C:/Users/tyler/Desktop/NSSSIP25/cvpr21_newt_pretrained_models.tar/cvpr21_newt_pretrained_models/tf/inat2021_mini/simclr_v2/#######_resnet50_simclr_v1_inat20_no_top.h5')

# Locking in the weights of the feature detection layers
model.trainable = False
for layer in model.layers:
	layer.trainable = False
    
# Split the images - do this once to avoid memory allocation issues!
# Running the functions to bring in our images and labels
trainimgs = Imagelist_resized
trainlbls = WP_io

trainimgs, testimgs, trainlbs, testlbls = train_test_split(Imagelist_resized,WP_io, test_size=0.2, random_state=69)

trainimgs_res = get_bottleneck_features(model, trainimgs)
testimgs_res = get_bottleneck_features(model, testimgs)

#%%
'''
Defining and training our classification NN: after passing through resnet50,
images are then passed through this network and classified. 
'''

# If we leave this code block seperate from the others, we can directly
# change our architecture and view the results

# Generate an input shape for our classification layers
input_shape = model.output_shape[1]

# Now we'll add new classification layers
model = Sequential()
model.add(InputLayer(input_shape = (input_shape,)))
model.add(Dense(256, activation = 'relu', input_dim = input_shape))
model.add(Dropout(0.5))
#model.add(Dense(128, activation = 'relu'))
#model.add(Dropout(0.3))
#model.add(Dense(64, activation = 'relu'))
#model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))

# Compiling our masterpiece
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-6), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])


"""
Let's train the model and take a look at how it does
"""
ne = 30
batch_size = 16
history = model.fit(trainimgs_res, trainlbs, 
                    validation_split = 0.25, 
                    epochs = ne, 
                    verbose = 1,
                    batch_size = batch_size,
                    shuffle=True)


#%%
'''
Visualization: inspect how the training went
'''
#model.save('ClassifierV1m.h5')
epoch_list = list(range(1,ne + 1))
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

#%% Implement some statistics

test_res = model.predict(testimgs_res)
test_res_binary = np.round(test_res)
#print(testlbls)
#print(test_res_binary.shape)
#print(len(testlbls))

# build out the components of a confusion matrix
n00, n01, n10, n11 = 0, 0, 0, 0 

for i, label_true in enumerate(testlbls):
    label_pred = test_res_binary[i]
    
    if label_true == 0:
        if label_pred == 0:
            n00 += 1
        if label_pred == 1:
            n01 += 1 
    elif label_true == 1:
        if label_pred == 0:
            n10 += 1
        if label_pred == 1:
            n11 += 1
       
n0 = n00 + n01
n1 = n10 + n11

# Compute accuracy, sensitivity, specificity, 
# positive prec, and neg prec
# As defined in:
    # Introducing Image Classification Efficacies, Shao et al 2021
    # or https://arxiv.org/html/2406.05068v1
    
    
acc = (n00 + n11) / len(testlbls) # complete accuracy
Se = n11 / n1 # true positive success rate
Sp = n00 / n0 # true negative success rate
Pp = n11 / (n11 + n01) # correct positive cases over all pred positive
Np = n00 / (n00 + n10) # correct negative cases over all pred negative


# Rate comapared to guessing
# MICE -> 1: perfect classification. -> 0: just guessing
A0 = (n0/len(testlbls))**2 + (n1/len(testlbls))**2
MICE = (acc - A0)/(1-A0)   

#%%
ntot = len(testlbls)
print("------------Test Results------------")
print("            Predicted Class         ")
print("True Class     0        1    Totals ")
print(f"     0        {n00}       {n01}    {n0}")
print(f"     1        {n10}        {n11}    {n1}")
print("")
print("            Predicted Class         ")
print("True Class     0        1    Totals ")
print(f"     0        {n00/ntot}      {n01/ntot}    {n0}")
print(f"     1        {n10/ntot}      {n11/ntot}    {n1}")
print("")
print(f"Model Accuracy: {acc}, Sensitivity: {Se}, Specificity: {Sp}")
print(f"True Positive rate: {Pp}, True Negative Rate: {Np}")
print(f"MICE (0->guessing, 1->perfect classification): {MICE}")
