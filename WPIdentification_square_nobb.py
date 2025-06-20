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

import matplotlib.pyplot as plt

import h5py

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import class_weight
#import cv2

#%% Define slice_width based on multiple of wavelength

bound_height = 16 #height of boundary layer in pixels - 16 for run33 in this case
num_wavelengths = 1.5 #number of wavelengths in each slice

slice_width = 2*bound_height * num_wavelengths 
print(f'slice_width = {slice_width}')
#left over parts of the image at the end are just discarded

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
	print('Getting Feature Data From ResNet...')
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
file_name = "C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\wavepacket_labels_combined.txt"
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
    
    for i in range(math.floor(num_slices)):
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
    model.add(Dense(256,                                        # NN dimension            
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
    ne = 20
    batch_size = 16
    history = model.fit(trainimgs_res, trainlbs, 
                        validation_split = 0.25, 
                        epochs = ne, 
                        verbose = 1,
                        batch_size = batch_size,
                        shuffle=True,
                        class_weight = class_weights_dict)
    
    # Return the results!
    # On this model, we need to return the processed test images for validation 
    # in the later step
    return history, model, testimgs_res, ne

#%% Train the fine tuning model

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
test_res= model.predict(testimgs_res)
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
model.save('C:\\Users\\Joseph Mockler\\Documents\\GitHub\\2025_Hypersonic_BL_ID\\ConeFlareRe33_normal.keras')

