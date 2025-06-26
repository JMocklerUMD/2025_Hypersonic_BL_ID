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

from matplotlib.patches import Rectangle

#%% Be able to run Second-Mode Wave detection, Turbulence detection, or both 
#(both defaults to using Second-Mode Wave detection dataset for labeling and whole-set statistics)

second_mode = True
sm_file_name = "C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\wavepacket_labels_combined.txt"
sm_N_img = 200
if second_mode:
    print('Finding second-mode waves')

#turbulence currently does not do post-processing
turb = True
turb_file_name = "C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\Test1\\run33\\turbulence_training_data.txt"
turb_N_img = 200
if turb:
    print('Finding turbulence')
    
whole_set_file_name = "C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\110000_111000_decimateby1\\Test1\\run33\\video_data.txt"

slice_width = 64
ne = 20
plot_flag = 0      # View the images? MUCH SLOWER (view - 1, no images - 0)
N_frames = -1      # Number of frames to go through for whole-set
                    # If you want the whole-set -> N_frames = -1

pro_speed_pix_frame = 43 # propagation speed in pixels/frame



if not second_mode and not turb:
    raise ValueError('One or both of "second_mode" and "turb" must be true')
    
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
    verbose = 1
    if input_imgs.shape == (224,224,3): #adds batch dimension for single images
        input_imgs = np.expand_dims(input_imgs, axis=0)                # Shape: (1, 224, 224, 3)
        verbose = 0
    if verbose == 1:
        print('Getting Feature Data From ResNet...')
    features = model.predict(input_imgs, verbose = verbose)
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

# Split the image into 20 pieces
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
    
    height, width = full_image.shape
    num_slices = width // slice_width
    
    # Only convert bounds to int if not sm_check.startswith('X')
    if not sm_check.startswith('X'):
        sm_bounds = list(map(int, sm_bounds))
        x_min, y_min, box_width, box_height = sm_bounds
        x_max = x_min + box_width
        y_max = y_min + box_height
    
    for i in range(num_slices-1):
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

def whole_image(i, lines):
    curr_line = i;
    line = lines[curr_line]
    
    parts = line.strip().split()
    
    image_size = list(map(int, parts[6:8]))  # Convert image size to integers
    image_data = list(map(float, parts[8:]))  # Convert image data to floats
    
    # Reshape the image data into the specified image size
    full_image = np.array(image_data).astype(np.float64)
    full_image = full_image.reshape(image_size)  # Reshape to (rows, columns)
    
    return full_image

def classify_the_images(model, resnet_model, Imagelist):
        Imagelist_resized = np.array([img_preprocess(img) for img in Imagelist])
        # Run through feature extractor
        Imagelist_res = get_bottleneck_features(resnet_model, Imagelist_resized)
        
        # Pass each through the trained NN
        test_res= model.predict(Imagelist_res, verbose = 0)
        classification_result = np.round(test_res)
        return classification_result, test_res, Imagelist_res


def classify_the_frame(Imagelist,WP_io, confidence, window_size, indiv_thres, model_turb,Imagelist_res):
    n00, n01, n10, n11 = 0, 0, 0, 0 
    filtered_result = []
    classification_result = np.zeros(len(Imagelist))
    
    if second_mode:
        for i, _ in enumerate(Imagelist):
            # If using the windowed post processing, call the windowing fcn
            # to get the locally informed confidence. Then compare to thresholds
            if use_post_process == 1:
                local_confid = calc_windowed_confid(i, confidence, window_size)
                
                # Are window and indiv conditions met?
                if (local_confid > confid_thres) or (confidence[i] > indiv_thres):
                    filtered_result.append(1)
                elif turb: #if we are also checking for turbulence
                    #Imagelist_resized = img_preprocess(Imagelist[i])
                    # Run through feature extractor
                    #Imagelist_res = get_bottleneck_features(resnet_model, Imagelist_resized)
                    test_res = model_turb.predict(Imagelist_res[i:i+1],verbose=0) #checks for turbulence
                    if test_res < 0:
                        filtered_result.append(0)
                    else:
                        filtered_result.append(2+test_res)
                else:
                    filtered_result.append(0)
                
                classification_result[i] = filtered_result[i]
            
            # If not, then just round
            else:
                check = np.round(confidence[i])
                if check == 0 and turb:
                    #Imagelist_resized = img_preprocess(Imagelist[i])
                    # Run through feature extractor
                    #Imagelist_res = get_bottleneck_features(resnet_model, Imagelist_resized)
                    test_res = model_turb.predict(Imagelist_res[i:i+1],verbose=0) #checks for turbulence
                    if test_res < 0.5:
                        classification_result[i] = 0
                    else:
                        classification_result[i] = 2+test_res
                elif check == 0:
                    classification_result[i] = 0
                else:
                    classification_result[i] = check
                
            # Get stats on the current image
            if WP_io[i] == 0:
                if classification_result[i] != 1:
                    n00 += 1
                if classification_result[i] == 1:
                    n01 += 1 
            elif WP_io[i] == 1:
                if classification_result[i] != 1:
                    n10 += 1
                if classification_result[i] == 1:
                    n11 += 1
    else: #for turbulence only
        for i, _ in enumerate(Imagelist):
            test_res = confidence[i]
            if test_res < 0.5:
                classification_result[i] = 0
            else:
                classification_result[i] = 2+test_res
            
            # Get stats on the current image
            if WP_io[i] == 0:
                if classification_result[i] == 0:
                    n00 += 1
                if classification_result[i] != 0:
                    n01 += 1 
            elif WP_io[i] == 1:
                if classification_result[i] == 0:
                    n10 += 1
                if classification_result[i] != 0:
                    n11 += 1                
            
                
    return classification_result, filtered_result, n00, n01, n10, n11

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


#%% Write training data to required arrays and visualize



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
        
        for i in range(num_slices-1):
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
    
    trainimgs, testimgs, trainlbs, testlbls = train_test_split(Imagelist_resized,WP_io, test_size=0.2, random_state=69)
    
    print("Done Splitting")
    
    return trainimgs, testimgs, trainlbs, testlbls, lines_len

#%% Split the test and train images
if second_mode:
    trainimgs, testimgs, trainlbs, testlbls, lines_len = write_data(sm_file_name, sm_N_img, slice_width)
if turb:
    trainimgs_turb, testimgs_turb, trainlbs_turb, testlbls_turb, lines_len_turb = write_data(turb_file_name, turb_N_img, slice_width)

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
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
                                    monitor='val_accuracy',
                                    min_delta=0,
                                    patience=5,
                                    verbose=0,
                                    mode='auto',
                                    restore_best_weights=True,
                                )
    
    # Train the model! Takes about 20 sec/epoch
    batch_size = 16
    history = model.fit(trainimgs_res, trainlbs, 
                        validation_split = 0.25, 
                        epochs = ne, 
                        verbose = 1,
                        batch_size = batch_size,
                        shuffle=True,
                        class_weight = class_weights_dict,
                        #callbacks = [early_stopping]
                        )
    
    # Return the results!
    # On this model, we need to return the processed test images for validation 
    # in the later step
    return history, model, testimgs_res, ne

#%% Finetuning model code was here

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
if second_mode:
    history, model, testimgs_res, ne = feature_extractor_training(trainimgs, trainlbs, testimgs)
    #history, model, testimgs_res, ne = feature_extractor_fine_tuning(trainimgs, trainlbs, testimgs)
    print("Second-mode Wave Model Training Complete!")

if turb:
    history_turb, model_turb, testimgs_res_turb, ne_turb = feature_extractor_training(trainimgs_turb, trainlbs_turb, testimgs_turb)
    print("Turbulence Model Training Complete!")

#%% Perform the visualization
'''
Visualization: inspect how the training went
'''
def stats(ne,history,model,testimgs_res,testlbls,name):
    #model.save('ClassifierV1m.h5')
    epoch_list = list(range(1,ne + 1))
    # Making some plots to show our results
    f, (pl1, pl2) = plt.subplots(1, 2, figsize = (15,4), gridspec_kw = {'wspace': 0.3})
    t = f.suptitle('Neural Network Performance: ' + name, fontsize = 14)
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
    
    # Implement some statistics
    
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
    
    # Print out the summary statistics
    ntot = len(testlbls)
    print("---------" + name + " Test Results---------")
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
    
#%% Show results
if second_mode:
    stats(ne,history,model,testimgs_res,testlbls,'Second-mode')

if turb:
    stats(ne_turb,history_turb,model_turb,testimgs_res_turb,testlbls_turb,'Turbulence')

#%% Save off the model, if desired
#model.save('C:\\Users\\tyler\\Desktop\\NSSSIP25\\TrainedModels\\run33_strangelyhighvalacc_95.keras')
#model_turb.save('C:\\Users\\tyler\\Desktop\\NSSSIP25\\TrainedModels\\turb_model_June24_bad.keras')
#print('Model(s) saved')
#%% add in classifer, post processing, and turbulence visualization code below...

#%% Classification code - stats only for file_name labels!!! (not other classes)

#%% Read image and existing classification model into the workspace
print('Reading training data file')

# Write File Name
if os.path.exists(whole_set_file_name):
    with open(whole_set_file_name, 'r') as file:
        lines = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len = len(lines)
print(f"{lines_len} lines read")

# Transfer learning model for stacking on ResNet50
model1 = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
output = model1.output
output = tf.keras.layers.Flatten()(output)
resnet_model = Model(model1.input,output)

#no need to load in model - already exists

#%% Iterate through the list!
# Initialize some arrays for later data analysis
N_img = lines_len
acc_history = []
TP_history = []
TN_history = []
FP_history = []
FN_history = []
WP_io_history = []
confidence_history = []
filtered_result_history = []
classification_history = []


plot_flag = plot_flag       # View the images? MUCH SLOWER
window_size = 3             # Moving window to filter the frames
indiv_thres = 0.85          # Individual exception threshold
confid_thres = 1.5          # SUMMED confidence over the entire window. 
                            # e.g. for 0.5 over 3 windows, make this value 1.5
use_post_process = 1        # 1 to use windowing post process, 0 if not

if N_frames == -1:
    N_frames = N_img

### Iterate over all frames in the video
for i_iter in range(N_frames): #range(N_img) can be changed to a range(#) for shorter loops for troubleshooting
    
    ### Perform the classification
    
    if not second_mode:
        model = 0 #useless input
    if not turb:
        model_turb = 0 #useless input to satisfy classify_the_images inputs if not finding turbulence
    
    # Split the image and classify the slices
    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines, slice_width)
        
    if second_mode:
        simple_class_result, confidence, Imagelist_res = classify_the_images(model, resnet_model, Imagelist)
    else:
        simple_class_result, confidence, Imagelist_res = classify_the_images(model_turb, resnet_model, Imagelist)
        
    # Analyze and filter the image results
    classification_result, filtered_result, n00, n01, n10, n11 = classify_the_frame(Imagelist,WP_io, confidence, window_size, indiv_thres,model_turb,Imagelist_res)
    
    
    ### Restitch and display the classification results
    # Restack and plot the image
    imageReconstruct = np.hstack([image for image in Imagelist])
    if plot_flag == 1:
        fig, ax = plt.subplots(1)
        ax.imshow(imageReconstruct, cmap = 'gray')
        
        if second_mode:
            ax.text(-45, height+86,'WP: ', fontsize = 6)
        if turb:
            ax.text(-57, height+62,'Turb: ', fontsize = 6)

        
        # Add on classification box rectangles
        for i, _ in enumerate(Imagelist):    
            # Add in the classification guess
            if classification_result[i] == 1:
                rect = Rectangle((i*slice_width, 5), slice_width, height-10,
                                         linewidth=0.5, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            elif classification_result[i] >= 2:
                ax.text(i*slice_width+slice_width/5, height+62,f'{(classification_result[i]-2):.2f}', fontsize = 6)
                if classification_result[i]-2 >= 0.5:
                    rect = Rectangle((i*slice_width, 5), slice_width, height-10,
                                             linewidth=0.5, edgecolor='orange', facecolor='none')
                    ax.add_patch(rect)
            if second_mode:
                prob = round(confidence[i,0],2)
                ax.text(i*slice_width+slice_width/5, height+86,f'{prob:.2f}', fontsize = 6)
            
                
        '''
            # Adds a rectangle for the confidence of classification at every square
            prob = confidence[i,0]
            rect = Rectangle((i*slice_width, 5), slice_width, height-10,
            linewidth=1.0*prob*prob, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(i*slice_width, height+60,round(prob,2), fontsize = 7)
            '''
            
            
    ### Save off the classification results for later analysis
    acc = (n00 + n11) / (n00 + n11 + n10 + n01)
    TP_history.append(n11)
    TN_history.append(n00)
    FP_history.append(n01)
    FN_history.append(n10)
    acc_history.append(acc)
    confidence_history.append(confidence)
    WP_io_history.append(WP_io)
    filtered_result_history.append(filtered_result)
    classification_history.append(classification_result)
    
    
    ### Finally, plot the ground truth
    if plot_flag == 1:
        # Check if there's even a bounding box in the image
        if sm_bounds[0] == 'X':
            ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN WP. Orange: NN Turbulence')
            plt.show()
            continue
        else:
            # Add the ground truth over the entire box
            ax.add_patch(Rectangle((sm_bounds[0],sm_bounds[1]), sm_bounds[2], sm_bounds[3], edgecolor='blue', facecolor='none'))
            if second_mode and turb:
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN WP. Orange: NN Turbulence')
            elif second_mode:
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN WP')
            else:
                ax.set_title('Image '+str(i_iter)+'. Blue: true Turbulence. Orange: NN Turbulence')
            plt.show()

print('Done classifying the video!')

#%% Whole-set stats
def whole_set_stats(Imagelist,acc_history,TP_history,TN_history,FP_history,FN_history):
#%% Make history plot
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
    
    
    # Compute MICE
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
    
    
    # Print out the entire data set statistics
    print("Data set statistics")
    print("----------------------------------------")
    print(f"Whole-set Average: {np.mean(acc_history)}")
    print(f"Whole-set True Positive rate: {np.mean(TP_history)/Nframe_per_img}")
    print(f"Whole-set True Negative rate: {np.mean(TN_history)/Nframe_per_img}")
    print(f"Whole-set False Positive rate: {np.mean(FP_history)/Nframe_per_img}")
    print(f"Whole-set False Negative rate: {np.mean(FN_history)/Nframe_per_img}")
    print(f"Whole-set MICE Score: {np.mean(MICE)}")
    
    
    # Form an ROC curve
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

#%% Run whole-set stats
if second_mode:
    print('Whole-set stats based on Second-mode Waves')
    whole_set_stats(Imagelist,acc_history,TP_history,TN_history,FP_history,FN_history)
else:
    print('Whole-set stats based on Turbulence')
    whole_set_stats(Imagelist,acc_history,TP_history,TN_history,FP_history,FN_history)

#%% Simple breakdown code
# if a slice is turbulence check if the slice before it in the previous frame was a WP
# assumes waves propogate at one slice per frame interval

where = [] #store where the breakdown occurs
    
if second_mode and turb: 
    ### Iterate over all frames in the video
    for i_iter in range(N_frames):
        if i_iter == 0: #skips the first frame (can't go back in time to check breakdown)
            continue
        for i, _ in enumerate(Imagelist):
            if i == 0: #skips the first slice (can't go back in space to check breakdown)
                continue
            if (classification_history[i_iter][i])-2 > 0.5: #if turbulent...
                if classification_history[(i_iter-1)][(i-1)]: #look back one in time and space to see it a WP preceeded it
                    where.append(i)
    plt.hist(where)
    plt.show()
    
#%% More complex breakdown code

#can maybe add code later to do unit propagation speed conversion from m/s or something comparable

thres = 0.5
turb_detect_count = 0
turb_count = 0
from_count = 0
dis_trav = []
preserve_classification_history = classification_history #saves a copy before overwriting later

if second_mode and turb: 
    #find total number of slices detected as turbulence
    for i_iter in range(N_frames-1,0,-1):
        for i, _ in enumerate(Imagelist):
            if (classification_history[i_iter][i])-2 > thres:
                turb_detect_count = turb_detect_count + 1 
                
    #find breakdown stats
    ### Iterate over all frames in the video
    for i_iter in range(N_frames-1,0,-1): #goes through frames backwards; stops before first image
        for i, _ in enumerate(reversed(Imagelist)): #goes through images starting on the right
            if i <= pro_speed_pix_frame//slice_width-1: #skips the first slice (or multiple if pro. speed is high enough) -- (can't go back in space to check breakdown)
                continue
            if (classification_history[i_iter][i])-2 > thres: #if turbulent...
                from_second_mode = False
                turb_count = turb_count + 1
                N_loop = 1
                while classification_history[(i_iter-N_loop)][(i-pro_speed_pix_frame*N_loop//slice_width)]-2 > thres: #check for preceeding turbulence
                    print('overwritten')
                    classification_history[(i_iter-N_loop)][(i-pro_speed_pix_frame*N_loop//slice_width)] = 0 #overwrite to 0 to avoid double counting turbulence
                    if i-pro_speed_pix_frame*N_loop//slice_width > 0 and i_iter-N_loop>=0: #check to avoid exceeding image bounds and first frame
                        N_loop = N_loop + 1
                    else:
                        patience = 0 
                        break
                mark = N_loop
                patience = 2 #gives second chance if WP is not detected the first time
                while patience >= 1:
                    if classification_history[(i_iter-N_loop)][(i-pro_speed_pix_frame*N_loop//slice_width)] == 1:
                        if i-pro_speed_pix_frame*N_loop//slice_width > 0 and i_iter-N_loop>=0: #check to avoid exceeding image bounds and first frame
                            N_loop = N_loop + 1
                        else:
                            break
                        if patience == 1:
                            patience = 2 #reset second chance if WP is detected
                        from_second_mode = True
                    else:
                        patience = patience - 1
                if from_second_mode:
                    from_count = from_count + 1
                if N_loop != mark: #checks if turbulence actually came from observed wave packet
                    dis_trav.append((N_loop-mark)*pro_speed_pix_frame)
    
    print(f'Total number of turbulence slices detected: {turb_detect_count}')           
    print(f'Turbulence count (duplicates overwritten): {turb_count}')
    print(f'Percentage of turbulence that developed from WPs: {round(from_count/turb_count*100,2)}%')
    print(f'Mean distance traveled: {round(np.mean(dis_trav),2)} pixels')
    print(f'Median distance traveled: {round(np.median(dis_trav),2)} pixels')
    plt.hist(dis_trav)
    plt.title('Distance WP traveled before breaking down into turbulence')
    plt.ylabel('Number of WPs')
    plt.xlabel('Distance traveled (pixels)')
    plt.show()
    