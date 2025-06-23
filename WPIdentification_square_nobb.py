# Version 2 of a CNN used to identify second-mode waves using ResNet50 for feature extraction

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from PIL import Image, ImageChops

import numpy as np
import os
import math
import random

import keras

import tensorflow as tf

from keras import optimizers, layers, regularizers

from keras.applications import resnet50, vgg16

from keras.callbacks import EarlyStopping

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer

from keras.models import Model
from keras.models import Sequential

from keras.preprocessing import image

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

import h5py

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import class_weight

from matplotlib.patches import Rectangle
#import cv2
import time
#%% Define slice_width based on multiple of wavelength

bound_height = 16 #height of boundary layer in pixels - 16 for run33 in this case
num_wavelengths = 2 #number of wavelengths in each slice

slice_width = int(2*bound_height * num_wavelengths)
print(f'slice_width = {slice_width}')
#left over parts of the image at the end are just discarded

ne = 20


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
	'''
    Retrives the ResNet50 feature vector
    INPUTS:  model:      resnet50 Keras model
             input_imgs: (N, 224, 224, 3) numpy array of (224, 224, 3) images to extract features from       
    OUTPUTS: featues:   (N, 100352) numpy array of extracted ResNet50 features
    '''
	#print('Getting Feature Data From ResNet...')
	features = model.predict(input_imgs, verbose = 1)
	return features

def img_preprocess(input_image):
    '''
    Reshapes the images for ResNet50 input
    INPUTS:  input_image:       (64,64) numpy array of double image data prescaled between [0,1]     
    OUTPUTS: processed_image:   (224, 224, 3) numpy array of the stretched input data
    '''
    input_image = np.stack((input_image,)*3,axis = -1)
    input_image = array_to_img(input_image)
    input_image = input_image.resize((224,224))
    processed_image = img_to_array(input_image)
    #input_image = (input_image / 127.5) - 1
    return processed_image


#%% Read training data file
# PREPROCESSING: the following block of codes accept the image data from a 
# big text file, parse them out, then process them into an array
# that can be passed to the keras NN trainer

print('Reading training data file')

# Write File Name
file_name = 'C://Users//tyler//Desktop//NSSSIP25//CROPPEDrun33//wavepacket_labels_combined.txt'
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
N_img, N_tot = 200, lines_len
print(f'N_img = {N_img}')
i_sample, img_count = 0, 0
sampled_list = []
discard_history = []

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
    
    #added to account for uncropped Langley run 34
    if full_image.shape == (64,1280):
        width = 1216

    #base slice_width on multiple on wavelength
    height, width = full_image.shape
    num_slices = width // slice_width
    
    discarded_length = width - (num_slices * slice_width)

    discard_history.append(discarded_length)
    #print(f'Discarded image {img_count} length = {discarded_length} pixels    Width  = {width}')
    
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
    
'''
print('while (img_count < N_img) and (i_sample < N_tot):')
print(f'img_count = {img_count}')
print(f'N_img = {N_img}')
print(f'i_sample = {i_sample}')
print(f'N_tot = {N_tot}')
'''
    
plt.plot(list(range(img_count)), discard_history)
plt.ylabel('Pixels Discarded')
plt.xlabel('Image')
plt.yticks(np.arange(0, max(discard_history) + 1, 4))
plt.title('Discarded Pixels at the End of Images')
plt.show()

avg_discarded = sum(discard_history)/len(discard_history)

print(f'Average discarded length = {avg_discarded} pixels')

print('Done sampling images!')


#%% Resizes the arrays
Imagelist = np.array(Imagelist)
WP_io = np.array(WP_io)
Imagelist_resized = np.array([img_preprocess(img) for img in Imagelist])
print("Done Resizing")

#%% Split the test and train images
trainimgs, testimgs, trainlbs, testlbls = train_test_split(Imagelist_resized,WP_io, test_size=0.2, random_state=69)
print("Done Splitting")

#%% Train the feature extractor model only

def feature_extractor_training(trainimgs, trainlbs, testimgs):
    """
    Building the Resnet50 model: a 256-dense NN is trained on ResNet50 features to classify the images.
    
    INPUTS: trainimgs:      (N, 224, 224, 3) numpy array of (224, 224, 3) image slices to train the model.
            trainlbs:       (N,1) numpy array of binary classes
            testimgs:       (M, 224, 224, 3) numpy array of (224, 224, 3) image slices to test the model.
    
    OUTPUTS: history:       keras NN model training history object
             model:         trained NN model of JUST the 256 dense NN
             testimgs_res:  (M, 100532) ResNet50 feature vector for each test image slice
             ne:            number of epochs trained
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
    
    
   
    #Defining and training our classification NN: after passing through resnet50,
    #images are then passed through this network and classified. 
    print('Extracting features...')
    trainimgs_res = get_bottleneck_features(resnet_model, trainimgs)
    testimgs_res = get_bottleneck_features(resnet_model, testimgs)
    
    # Generate an input shape for our classification layers
    input_shape = resnet_model.output_shape[1]
    
    # Get unique classes and compute weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(trainlbs),
        y=trainlbs
        )
    
    # Convert to dictionary format required by Keras
    class_weights_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weights_dict)
    
    # Added the classification layers
    model = Sequential()
    model.add(InputLayer(input_shape = (input_shape,)))
    model.add(Dense(128,                                        # NN dimension            
                    activation = 'relu',                        # Activation function at each node
                    input_dim = input_shape,                    # Input controlled by feature vect from ResNet50
                    kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4),     # Regularization penality term
                    bias_regularizer=regularizers.L2(1e-4)))                    # Additional regularization penalty term
    
    model.add(Dropout(0.5))     # Add dropout to make the system more robust
    model.add(Dense(1, activation = 'sigmoid'))     # Add final classification layer
    
    # Compile the NN
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-6), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy'])
    
    # Inspect the resulting model
    model.summary()
    
    # Train the model! Takes about 20 sec/epoch
    batch_size = 16
    
    start_time = time.time()
    
    history = model.fit(trainimgs_res, trainlbs, 
                        validation_split = 0.25, 
                        epochs = ne, 
                        verbose = 1,
                        batch_size = batch_size,
                        shuffle=True,
                        class_weight = class_weights_dict)
    
    train_time = time.time() - start_time
    print(f'Training/fitting time: {train_time} seconds')
    
    # Return the results!
    # On this model, we need to return the processed test images for validation 
    # in the later step
    return history, model, testimgs_res, ne

#%% Call fcn to train the model!
history, model, testimgs_res, ne = feature_extractor_training(trainimgs, trainlbs, testimgs)
#history, model, testimgs_res, ne = feature_extractor_fine_tuning(trainimgs, trainlbs, testimgs)
print("Training Complete!")

#%% Perform the visualization
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

# Check how well we did on the test data!
test_res= model.predict(testimgs_res, verbose = 0)
test_res_binary = np.round(test_res)

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
    # or https://neptune.ai/blog/evaluation-metrics-binary-classification
    
TP = n11
TN = n00
FP = n01
FN = n10
    
acc = (n00 + n11) / len(testlbls) # complete accuracy
Se = n11 / n1 # true positive success rate, recall
Sp = n00 / n0 # true negative success rate
Pp = n11 / (n11 + n01) # correct positive cases over all pred positive
Np = n00 / (n00 + n10) # correct negative cases over all pred negative
Recall = TP/(TP+FN) # Probability of detection
FRP = FP/(FP+TN) # False positive, probability of a false alarm

# Rate comapared to guessing
# MICE -> 1: perfect classification. -> 0: just guessing
A0 = (n0/len(testlbls))**2 + (n1/len(testlbls))**2
MICE = (acc - A0)/(1-A0)   

#%% Print out the summary statistics
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
print(f"Precision: {Pp},  Recall: {Recall}, False Pos Rate: {FRP}")
print(f"MICE (0->guessing, 1->perfect classification): {MICE}")
print("")
print(f"True Pos: {n11}, True Neg: {n00}, False Pos: {n01}, False Neg: {n10}")


#%% Save off the model, if desired
#model.save('C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\ConeFlareRe33_normal.keras')

#%% Train the fine tuning model

#%% Classification Code
print('Running classification code')

#%% Function Calls + Resnet50 instantiation
# This function tells our feature extractor to do its thing
def get_bottleneck_features(model, input_imgs):
	#print('Getting Feature Data From ResNet...')
	features = model.predict(input_imgs, verbose = 0)
	return features

def img_preprocess(input_image):
    input_image = np.stack((input_image,)*3,axis = -1)
    input_image = array_to_img(input_image)
    input_image = input_image.resize((224,224))
    input_image = img_to_array(input_image)
    #input_image = (input_image / 127.5) - 1
    return input_image


model1 = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
output = model1.output
output = tf.keras.layers.Flatten()(output)
resnet_model = Model(model1.input,output)

# load the classifier
# model = keras.models.load_model('C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\ConeFlareRe33_normal.keras')


#%% read in images
print('Reading training data file')

# Write File Name
file_name = "C:\\Users\\tyler\\Desktop\\NSSSIP25\\LangleyM6_Run34.txt"
if os.path.exists(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len = len(lines)
print(f"{lines_len} lines read")


#%% Split the image into 20 pieces
def image_splitting(i, lines, slice_width):
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
        
    #added to account for uncropped Langley run 34
    if full_image.shape == (64,1280):
        width = 1216    
    
    #slice_width = 64
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
                
    return Imagelist, WP_io, slice_width, height, sm_bounds

def classify_the_images(model, Imagelist):
    Imagelist_resized = np.array([img_preprocess(img) for img in Imagelist])
    
    # Run through feature extractor
    Imagelist_res = get_bottleneck_features(resnet_model, Imagelist_resized)
    
    # Pass each through the trained NN
    test_res= model.predict(Imagelist_res, verbose = 0)
    classification_result = np.round(test_res)
    
    return classification_result, test_res


#%% Iterate through the list!
N_img = lines_len
print(f'N_img = {N_img}')
acc_history = []
TP_history = []
TN_history = []
FP_history = []
FN_history = []
WP_io_history = []
confidence_history = []
plot_flag = 0       # View the images? MUCH SLOWER

print('Finding image accuracy (this may take a while)...')
for i_iter in range(N_img):
    
    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines, slice_width)
    
    classification_result, confidence = classify_the_images(model, Imagelist)
  
    # Restack and plot the image
    imageReconstruct = np.hstack([image for image in Imagelist])
    
    if plot_flag == 1:
        fig, ax = plt.subplots(1)
        ax.imshow(imageReconstruct, cmap = 'gray')
    
    # build out the components of a confusion matrix
    n00, n01, n10, n11 = 0, 0, 0, 0 
               
    # Add on classification box rectangles
    for i, _ in enumerate(Imagelist):
        
        # Get stats on the current image
        if WP_io[i] == 0:
            if classification_result[i] == 0:
                n00 += 1
            if classification_result[i] == 1:
                n01 += 1 
        elif WP_io[i] == 1:
            if classification_result[i] == 0:
                n10 += 1
            if classification_result[i] == 1:
                n11 += 1
        
        # Add in the classification guess
        if classification_result[i] == 1 and plot_flag == 1:
            rect = Rectangle((i*slice_width, 5), slice_width, height-10,
                                     linewidth=0.5, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        if plot_flag == 1:
            # Adds a rectangle for the confidence of classification at every square
            prob = confidence[i,0]
            rect = Rectangle((i*slice_width, 5), slice_width, height-10,
            linewidth=1.0*prob*prob, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(i*slice_width, height+60,round(prob,2), fontsize = 7)
            
            
    # Compute the inter-image accuracy
    acc = (n00 + n11) / (n00 + n11 + n10 + n01)
    #print(f'Image {i_iter}: accuracy = {acc}')
    
    # Save off data for whole-set analysis
    TP_history.append(n11)
    TN_history.append(n00)
    FP_history.append(n01)
    FN_history.append(n10)
    acc_history.append(acc)
    confidence_history.append(confidence)
    WP_io_history.append(WP_io)
    
    if plot_flag == 1:
        # Check if there's even a bounding box in the image
        if sm_bounds[0] == 'X':
            ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN class')
            plt.show()
            continue
        else:
            # Add the ground truth over the entire box
            ax.add_patch(Rectangle((sm_bounds[0],sm_bounds[1]), sm_bounds[2], sm_bounds[3], edgecolor='blue', facecolor='none'))
        
            ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN class')
            plt.show()

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


#%% Compute MICE
Nframe_per_img = len(Imagelist)
MICE = []
for i in range(len(acc_history)):
    n0 = TP_history[i] + FP_history[i]
    n1 = TN_history[i] + FN_history[i]
    
    A0 = (n0/Nframe_per_img)**2 + (n1/Nframe_per_img)**2
    if np.isclose(1-A0, 0):
        MICE.append(0.0)
    else:
        MICE.append((acc_history[i] - A0)/(1-A0))
    
    
fig, ax = plt.subplots(1,1, figsize = (16,6))
ax.plot(range(len(MICE)), MICE)
ax.plot(range(n-1, len(MICE)), moving_average(MICE, n), color='k', linewidth = 2)
ax.set_ylim(-1,1)
ax.set_title('MICE Performance')


#%% Print out the entire data set statistics
print("Data set statistics")
print("----------------------------------------")
print(f"Whole-set Average: {np.mean(acc_history)}")
print(f"Whole-set True Positive rate: {np.mean(TP_history)/Nframe_per_img}")
print(f"Whole-set True Negative rate: {np.mean(TN_history)/Nframe_per_img}")
print(f"Whole-set False Positive rate: {np.mean(FP_history)/Nframe_per_img}")
print(f"Whole-set False Negative rate: {np.mean(FN_history)/Nframe_per_img}")
print(f"Whole-set MICE Score: {np.mean(MICE)}")


#%% Form an ROC curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
thresholds = np.linspace(0, 1, num=50)
TPRs, FPRs, Pres = [], [], []
# Loop thru the thresholds
for threshold in thresholds:
    TP, FP, TN, FN = 0, 0, 0, 0
    
    # Loop thru each image in the test set
    for i in range(len(acc_history)):
        
        # Pull off the sliced list
        WP_io_img = WP_io_history[i]
        confid_img = confidence_history[i]
        slice_classification = []
        
        # Form the classification list under the new thrshold
        n00, n01, n10, n11 = 0, 0, 0, 0 
        for j in range(len(WP_io_img)):
            if confid_img[j] > threshold:
                slice_classification.append(1)
            else:
                slice_classification.append(0)
                
            # Now compute the TPR/FPR of the frame
            if WP_io_img[j] == 0:
                if slice_classification[j] == 0:
                    n00 += 1
                if slice_classification[j] == 1:
                    n01 += 1 
            elif WP_io_img[j] == 1:
                if slice_classification[j] == 0:
                    n10 += 1
                if slice_classification[j] == 1:
                    n11 += 1
        
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
    

# Compute the AUC of the ROC - simple rectangular integration
AUC = 0.0
for i in range(1,len(TPRs)):
    AUC = AUC + (FPRs[i-1]-FPRs[i])*(TPRs[i]+TPRs[i-1])/2    
print(f'Area under the ROC Curve = {AUC}')

PR = 0.0
for i in range(1,len(TPRs)):
    PR = PR + (Pres[i]+Pres[i-1])*(TPRs[i-1]-TPRs[i])/2    
print(f'Area under the PR Curve = {PR}')

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




#%% Joe's post-processing code
print('Running post-processing')
#%% Load classification results
#WP_io = np.load('C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\True_class_Run38.npy')
#Confidence_history = np.load('C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\Confidence_class_Run38.npy')

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
        Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines, slice_width)
        confidence = confidence_history[i_iter]
        
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
Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(1, lines, slice_width)
N_img = lines_len
print(f'N_img = {N_img}')
plot_flag = 0               # View the images? MUCH SLOWER
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
    TP_history_post, TN_history_post, FP_history_post, FN_history_post, acc_history_post = [], [], [], [], []
    for i_iter in range(N_img): 
        
        # Split up image and get the labelled confidence
        Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines, slice_width)
        confidence = confidence_history[i_iter]
        
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
        TP_history_post.append(n11)
        TN_history_post.append(n00)
        FP_history_post.append(n01)
        FN_history_post.append(n10)
        acc_history_post.append(acc)
        
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
pl1.plot(range(len(acc_history_post)), acc_history_post)
pl1.plot(range(n-1, len(acc_history_post)), moving_average(acc_history_post, n), color='k', linewidth = 2)
pl1.set_title('Accuracy')

pl2.plot(range(len(TP_history_post)), [img_stat/Nframe_per_img for img_stat in TP_history_post])
pl2.plot(range(n-1, len(acc_history_post)), moving_average(TP_history_post, n)/Nframe_per_img, color='k', linewidth = 2)
pl2.set_title('True positive rate')

pl3.plot(range(len(TN_history_post)), [img_stat/Nframe_per_img for img_stat in TN_history_post])
pl3.plot(range(n-1, len(TN_history_post)), moving_average(TN_history_post, n)/Nframe_per_img, color='k', linewidth = 2)
pl3.set_title('True negative rate')

pl4.plot(range(len(FP_history_post)), [img_stat/Nframe_per_img for img_stat in FP_history_post])
pl4.plot(range(n-1, len(FP_history_post)), moving_average(FP_history_post, n)/Nframe_per_img, color='k', linewidth = 2)
pl4.set_title('False positive rate')

pl5.plot(range(len(FN_history_post)), [img_stat/Nframe_per_img for img_stat in FN_history_post])
pl5.plot(range(n-1, len(FN_history_post)), moving_average(FN_history_post, n)/Nframe_per_img, color='k', linewidth = 2)
pl5.set_title('False negative rate')

#%% Print out the entire data set statistics
print("Data set statistics")
print("----------------------------------------")
print(f"Whole-set Average: {np.mean(acc_history_post)}")
print(f"Whole-set True Positive rate: {np.mean(TP_history_post)/Nframe_per_img}")
print(f"Whole-set True Negative rate: {np.mean(TN_history_post)/Nframe_per_img}")
print(f"Whole-set False Positive rate: {np.mean(FP_history_post)/Nframe_per_img}")
print(f"Whole-set False Negative rate: {np.mean(FN_history_post)/Nframe_per_img}")

#%% Compare non-post-processed and post-processed results

print("------------Whole-set Test Results------------")
print("                   Non-processed         Post-processed         Difference")
print(f"Avg. Accuracy:         {round(np.mean(acc_history)/Nframe_per_img,3)}               {round(np.mean(acc_history_post)/Nframe_per_img,3)}          {-round(np.mean(acc_history)/Nframe_per_img-np.mean(acc_history_post)/Nframe_per_img,3)}")
print(f"True Positive:         {round(np.mean(TP_history)/Nframe_per_img,3)}               {round(np.mean(TP_history_post)/Nframe_per_img,3)}          {-round(np.mean(TP_history)/Nframe_per_img-np.mean(TP_history_post)/Nframe_per_img,3)}")
print(f"True Negative:         {round(np.mean(TN_history)/Nframe_per_img,3)}               {round(np.mean(TN_history_post)/Nframe_per_img,3)}          {-round(np.mean(TN_history)/Nframe_per_img-np.mean(TN_history_post)/Nframe_per_img,3)}")
print(f"False Positive:        {round(np.mean(FP_history)/Nframe_per_img,3)}               {round(np.mean(FP_history_post)/Nframe_per_img,3)}          {-round(np.mean(FP_history)/Nframe_per_img-np.mean(FP_history_post)/Nframe_per_img,3)}")
print(f"False Negative:        {round(np.mean(FN_history)/Nframe_per_img,3)}               {round(np.mean(FN_history_post)/Nframe_per_img,3)}          {-round(np.mean(FN_history)/Nframe_per_img-np.mean(FN_history_post)/Nframe_per_img,3)} ")
#%% Create video statistics plots
'''
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
'''
'''
def feature_extractor_fine_tuning(trainimgs, trainlbs, testimgs):
    """
    Building the Resnet50 model: a 256-dense NN and the top layers of the ResNet50 model are all trained
    
    INPUTS: trainimgs:      (N, 224, 224, 3) numpy array of (224, 224, 3) image slices to train the model.
            trainlbs:       (N,1) numpy array of binary classes
            testimgs:       (M, 224, 224, 3) numpy array of (224, 224, 3) image slices to test the model.
    
    OUTPUTS: history:       keras NN model training history object
             model:         trained NN model of JUST the 256 dense NN
             testimgs_res:  (M, 100532) ResNet50 feature vector for each test image slice
             ne:            number of epochs trained
    """
    # Form the base model
    base_model = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
    inputs = keras.Input(shape=(224,224,3))
    
    # Check length of model layers, if desired
    # print(len(base_model.layers))
    
    # Choose which layers to kept frozen or unfrozen
    for layer in base_model.layers[:155]: # the first 155 layers
        layer.trainable = False 
    
    # Construct the architecture
    x = inputs                                          # Start with image input
    x = base_model(x)                                   # pass thru Resnet50
    x = Flatten()(x)                                    # Flatten (just like above!)
    x = layers.Dense(256, activation = 'relu')(x)       # Pass thru the dense 256 arch
    x = layers.Dropout(0.5)(x)                          # Add dropout
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Final classification layer
    
    # Compile and train the model
    model_FineTune = Model(inputs, outputs)
    model_FineTune.summary()
    model_FineTune.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-6), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy']) # keep a low learning rate
    
    # Perform training. NOTE: takes around 4 min/epoch so be careful!
    ne = 20
    batch_size = 16
    history = model_FineTune.fit(trainimgs, trainlbs, 
                        validation_split = 0.25, 
                        epochs = ne, 
                        verbose = 1,
                        batch_size = batch_size,
                        shuffle=True)
    
    # Return the results!
    # On this model, we only need to return the testimages because we're NOT
    # running them thru the bottleneck first
    return history, model_FineTune, testimgs, ne

'''