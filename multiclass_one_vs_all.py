#%% Import libraries
import numpy as np
import os
import math
import random
import copy
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import tensorflow as tf

import keras
from keras import optimizers, layers, regularizers
from keras.applications import resnet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

print('Done importing libraries')
#%% Input files and run settings
print('Initializing file names and settings')

N_positive_cls = 1
file_names = []
class_names = []
N_imgs_list = []   # number of images to use for training and testing

if N_positive_cls >= 1:
    file_names.append("C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\wavepacket_labels_combined.txt")
    class_names.append('Second-mode Wave Packets')
    N_imgs_list.append(200) 
if N_positive_cls >= 2:
    file_names.append("C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\Test1\\run33\\turbulence_training_data.txt")
    class_names.append('Turbulence')
    N_imgs_list.append(10)
if N_positive_cls >= 3:
    file_names.append('')
    class_names.append('')
    N_imgs_list.append(200)
if N_positive_cls >= 4:
    file_names.append('')
    class_names.append('')
    N_imgs_list.append(200)
if N_positive_cls >= 5:
    file_names.append('')
    class_names.append('')
    N_imgs_list.append(200)
if N_positive_cls >= 6:
    file_names.append('')
    class_names.append('')
    N_imgs_list.append(200)

ne = 20            # Number of epoches
slice_width = 96
thres = 0.5        # Threshold for classifying an individual slice

use_early_stopping = True
metric = 'val_accuracy'         #val_loss, val_accuracy, etc
patience = 5

augment = False

whole_set_file_name = "C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\110000_111000_decimateby1\\Test1\\run33\\video_data.txt"
plot_flag = 0       # View the images? MUCH SLOWER (view - 1, no images - 0)
N_frames = -1       # Number of frames to go through for whole-set
                    # If you want the whole-set -> N_frames = -1

FR = 258e3                 # Camera frame rate in Hz
prop_speed = 825       # A priori estimate of propagation speed, m/s

#%% Error check and calculations from settings or constants
if N_positive_cls < 1 and  N_positive_cls > 6 and not isinstance(N_positive_cls,int):
    raise ValueError('N_positive_cls must be an integer 1 to 6')
    
# Calculate approx how many pixels a wave will propagate in a single frame
mm_pix = 0.0756        # From paper, mm to pixel conversion
dt = 1/FR                     # Time step between frames
prop_speed_pix_frame = prop_speed * dt * 1/(mm_pix*1e-3)  # Computed number of pixels traveled between frames

print('Done with constant initialization')
#%% Define functions
def img_preprocess(input_image):
    '''
    Reshapes the images for ResNet50 input
    INPUTS:  input_image:       (height,slice_width) numpy array of double image data prescaled between [0,1]     
    OUTPUTS: processed_image:   (224, 224, 3) numpy array of the stretched input data
    '''
    input_image = np.stack((input_image,)*3,axis = -1)
    input_image = array_to_img(input_image) #scales to 0-255
    input_image = input_image.resize((224,224))
    processed_image = img_to_array(input_image)
    processed_image = preprocess_input(processed_image) #resnet50 preprocessing
    return processed_image

def whole_image(i, lines):
    curr_line = i;
    line = lines[curr_line]
    
    parts = line.strip().split()
    
    image_size = list(map(int, parts[6:8]))  # Convert image size to integers
    image_data = list(map(float, parts[8:]))  # Convert image data to floats
    
    # Reshape the image data into the specified image size
    full_image = np.array(image_data).astype(np.float64)
    full_image = full_image.reshape(image_size)  # Reshape to (rows, columns)
    height, width = full_image.shape

    
    return full_image, height

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
        full_image = np.array(image_data).astype(np.float32)
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
    
    trainimgs, tempimgs, trainlbs, templbs = train_test_split(Imagelist_resized,WP_io, test_size=0.3, random_state=42)
    valimgs, testimgs, vallbs, testlbs = train_test_split(tempimgs,templbs, test_size=0.5, random_state=42)

    print("Done Splitting")
    
    return trainimgs, testimgs, valimgs, vallbs, trainlbs, testlbs, lines_len, img_count, num_slices, height

def write_data_whole(file_name, slice_width):
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

    print('Reading whole-set data file')

    # Write File Name
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()
    else:
        raise ValueError("No whole-set file detected")

    lines_len = len(lines)
    print(f"{lines_len} lines read")
    
    print('Begin writing whole-set data to numpy array')
    
    WP_io = []
    #SM_bounds_Array = []
    Imagelist_raw = []
    N_tot = lines_len
    i_sample, img_count = 0, 0
    sampled_list = []
    
    # Break when we aquire 100 images or when we run thru the 1000 frames
    while i_sample < N_tot:
    
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
        full_image = np.array(image_data).astype(np.float32)
        full_image = full_image.reshape(image_size)  # Reshape to (rows, columns)
        
        if full_image.shape == (64, 1280):
            width = 1216 #account for uncropped run34 images
        
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
            Imagelist_raw.append(image)
        
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
    print(f'Number of used images: {img_count}')
    print(f'slice_width  = {slice_width}')
    
    print('Done sampling images!')
    
    # Resizes the arrays
    Imagelist = np.array(Imagelist_raw)
    WP_io = np.array(WP_io)
    Imagelist = np.array([img_preprocess(img) for img in Imagelist])
    print("Done Resizing")
        
    return Imagelist, Imagelist_raw, WP_io, lines_len, img_count, num_slices, height

def feature_extractor_training(train_ds,val_ds,class_weights_dict,use_early_stopping,metric,patience):
    """
    Building the Resnet50 model: a 256-dense NN is trained on ResNet50 features to classify the images.
    
    INPUTS: train_ds:       dataset (images and corresponding labels) for training
            val_ds:         dataset (images and corresponding labels) for validation
    
    OUTPUTS: history:       keras NN model training history object
             model:         trained NN model of JUST the 256 dense NN
             fe:            number of epochs trained; best epoch if early stopping is enabled
    """

    # Bringing in ResNet50 to use as our feature extractor
    resnet_model = resnet50.ResNet50(include_top = False, weights ='imagenet')
    #GlobalAveragePooling2D() is a good alternative to flatten

    # Locking in the weights of the feature detection layers
    resnet_model.trainable = False
    for layer in resnet_model.layers:
    	layer.trainable = False
    
    # Added the classification layers
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))    
    model.add(resnet_model)
    model.add(Flatten())   #GlobalAveragePooling2D() is a good alternative to Flatten()
    model.add(Dense(128,                                        # NN dimension            
                    activation = 'relu'                         # Activation function at each node
                    #reg values originally 1e-4
                    ,kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4),     # Regularization penality term
                    bias_regularizer=regularizers.L2(1e-4)                     # Additional regularization penalty term
                    ))                   
    
    model.add(Dropout(0.5))     # Add dropout to make the system more robust
    model.add(Dense(1, activation = 'sigmoid'))     # Add final classification layer
    
    # Compile the NN
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-6), #originally 1e-6
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy'])
    
    # Inspect the resulting model
    model.summary()
    
    
    #add early stopping if use_early_stopping is True
    callbacks = []
    if use_early_stopping:
        early_stopping = tf.keras.callbacks.EarlyStopping(
                                        monitor=metric,
                                        min_delta=0,
                                        patience=patience,
                                        verbose=0,
                                        mode='auto',
                                        restore_best_weights=True,
                                    )
        callbacks.append(early_stopping)
    
    start = time.time()
    
    # Train the model!
    history = model.fit(train_ds, 
                        validation_data = val_ds, 
                        epochs = ne, 
                        verbose = 1,
                        shuffle=True,
                        class_weight = class_weights_dict,
                        callbacks = callbacks
                        )
    
    end = time.time()
    print(f'Model training time: {end-start} seconds')
    
    if use_early_stopping and early_stopping.stopped_epoch != 0: #second part of conditional checks to see if early stopping actually activiated
        fe = early_stopping.stopped_epoch + 1
        be = early_stopping.stopped_epoch - early_stopping.patience
    else:
        fe = len(history.history['loss']) 
        be = fe
    
    # Return the results!
    # On this model, we need to return the processed test images for validation 
    # in the later step
    return history, model, fe, be

def compute_class_weights(testlbs):
    # Get unique classes and compute weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(trainlbs),
        y=trainlbs
        )
    
    # Convert to dictionary format required by Keras
    class_weights_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weights_dict)
    return class_weights_dict

def model_stats(fe,history,model,test_ds,name,thres,feature_prop_speed,num_slices):
    #model.save('ClassifierV1m.h5')
    epoch_list = list(range(1,fe+1))
    # Making some plots to show our results
    f, (pl1, pl2) = plt.subplots(1, 2, figsize = (15,4), gridspec_kw = {'wspace': 0.3})
    t = f.suptitle('Neural Network Performance: ' + name, fontsize = 14)
    # Accuracy Plot
    pl1.plot(epoch_list, history.history['accuracy'], label = 'train accuracy')
    pl1.plot(epoch_list, history.history['val_accuracy'], label = 'validation accuracy')
    pl1.set_xticks(np.arange(0, fe+1, 5))
    pl1.set_xlabel('Epoch')
    pl1.set_ylabel('Accuracy')
    pl1.set_title('Accuracy')
    leg1 = pl1.legend(loc = "best")
    # Loss plot for classification
    pl2.plot(epoch_list, history.history['loss'], label = 'train loss')
    pl2.plot(epoch_list, history.history['val_loss'], label = 'validation loss')
    pl2.set_xticks(np.arange(0, fe+1, 5)) 
    pl2.set_xlabel('Epoch')
    pl2.set_ylabel('Loss')
    pl2.set_title('Classification Loss')
    leg2 = pl2.legend(loc = "best")
    plt.show()
    
    # Implement some statistics
    
    # Check how well we did on the test data!
    test_res = model.predict(test_ds)
    
    label_true = tf.concat([lbs for imgs, lbs in test_ds], axis=0)

    confusion_matrix(test_res,label_true,thres,name)
    #no returns

def confusion_matrix(label_pred,label_true,thres,name):
    # build out the components of a confusion matrix
    n00, n01, n10, n11 = 0, 0, 0, 0 
            
    for n,label_true in enumerate(label_true):
        if label_true == 0:
            if label_pred[n] < thres:
                n00 += 1
            if label_pred[n] >= thres:
                n01 += 1 
        elif label_true == 1:
            if label_pred[n] < thres:
                n10 += 1
            if label_pred[n] >= thres:
                n11 += 1
           
    n0 = n00 + n01
    n1 = n10 + n11
    
    tot_len = n0 + n1
    
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
        
    acc = (n00 + n11) / tot_len # complete accuracy
    Se = n11 / n1 # true positive success rate, recall
    Sp = n00 / n0 # true negative success rate
    Pp = n11 / (n11 + n01) # correct positive cases over all pred positive
    Np = n00 / (n00 + n10) # correct negative cases over all pred negative
    Recall = TP/(TP+FN) # Probability of detection
    FRP = FP/(FP+TN) # False positive, probability of a false alarm
    
    # Rate comapared to guessing
    # MICE -> 1: perfect classification. -> 0: just guessing
    A0 = (n0/tot_len)**2 + (n1/tot_len)**2
    MICE = (acc - A0)/(1-A0)   
    
    # Print out the summary statistics
    ntot = tot_len
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
    #no returns

def make_dataset(images, labels, batch_size=16, shuffle=False, augment_fn=None):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(1024)
    if augment_fn:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def augment_image(image, label):
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def post_process_time(confidence_history,feature_prop_speed,slice_width,num_slices):
    slice_move = feature_prop_speed // slice_width
    processed_conf_hist = copy.deepcopy(confidence_history)
    
    for n, confidence in enumerate(confidence_history):
        if n < num_slices or n>=len(confidence_history)-num_slices-slice_move:
            continue #leaves first and last frames alone & prevents overrunning the end of the array
        elif n%num_slices < slice_move or n%num_slices > num_slices - slice_move - 2:
            continue #leaves the first and last slices (or more/less with high/low prop speed) alone
        else:
            confidence = (confidence + confidence_history[int(n-num_slices-slice_move)] + confidence_history[int(n+num_slices+slice_move)]) / 2
            processed_conf_hist[n] = confidence
        
    return processed_conf_hist

def calc_windowed_confid(j, confidence_history, window_size,num_slices,i_iter):
    '''
    Calculates the local confidence (i.e. a single slice of a frame) 
    via a summed windowing method
    '''
    #offset for indexing
    os = i_iter * num_slices
    if (j - window_size//2) < 0: # at the front end of the image
        local_confid = np.sum(confidence_history[0+os:j+window_size//2+1+os:1+os])
    elif (j + window_size//2) > num_slices: # at the end of the image list
        local_confid = np.sum(confidence_history[j-window_size//2-1+os:num_slices+os:1+os])
    else:
        local_confid = np.sum(confidence_history[j-window_size//2+os:j+window_size//2+1+os:1+os])
        
    return local_confid

def visualize_frames(lines_len,N_frames,num_slices,Imagelist_raw,confidence_history,processed_conf_hist,thres,slice_width):
    if N_frames == -1:
        N_frames = lines_len
    print(f'Visualizing {N_frames} images (this may take a while)...')
    
    Imagelist_raw = [image / 255.0 for image in Imagelist_raw] #normalize to 0-1

    ### Iterate over all frames in the video
    for i_iter in range(N_frames):
        imageReconstruct = np.hstack(Imagelist_raw[(i_iter*num_slices):((i_iter+1)*num_slices)])
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.set_title('Original Classifications')
        ax1.imshow(imageReconstruct, cmap = 'gray')
        ax2.set_title('With Time Post-processing')
        ax2.imshow(imageReconstruct, cmap = 'gray')

        ax1.text(-120, height+86,'Confidence: ', fontsize = 6)
        ax2.text(-120, height+86,'Confidence: ', fontsize = 6)

        r_confid_hist = np.round(confidence_history,2)
        r_proc_conf_hist = np.round(processed_conf_hist,2)

        # Add on classification box rectangles
        for i, _ in enumerate(Imagelist_raw[(i_iter*num_slices):((i_iter+1)*num_slices)]):    
                # Add in the classification guess
                if confidence_history[i+i_iter*num_slices]>=thres:
                    rect = Rectangle((i*slice_width, 5), slice_width, height-10,
                                             linewidth=0.5, edgecolor='red', facecolor='none')
                    ax1.add_patch(rect)
                if processed_conf_hist[i+i_iter*num_slices]>=thres:
                    rect = Rectangle((i*slice_width, 5), slice_width, height-10,
                                             linewidth=0.5, edgecolor='red', facecolor='none')
                    ax2.add_patch(rect)
                ax1.text(i*slice_width+slice_width/6, height+86,r_confid_hist[i+i_iter*num_slices], fontsize = 6)
                ax2.text(i*slice_width+slice_width/6, height+86,r_proc_conf_hist[i+i_iter*num_slices], fontsize = 6)
        fig.suptitle(f'Image {i_iter}')
        plt.show()
    print('Done visualizing the frames')


print('Done defining functions')

#%% Split the test and train images

lines_len_hist = []
used_img_count_hist = []
history_hist = []
model_hist = []
fe_hist = []
be_hist = []

if augment:
    augment_fn = augment_image
else:
    augment_fn = None

for n in range(N_positive_cls):
    print(f'Training model for {class_names[n]}')
    trainimgs, testimgs, valimgs, vallbs, trainlbs, testlbs, lines_len, used_img_count, num_slices, height = write_data(file_names[n], N_imgs_list[n], slice_width)
    class_weights_dict = compute_class_weights(trainlbs)
    train_ds = make_dataset(trainimgs, trainlbs, shuffle=True, augment_fn=augment_fn)
    val_ds   = make_dataset(valimgs, vallbs)
    test_ds  = make_dataset(testimgs, testlbs)
    history, model, fe, be = feature_extractor_training(train_ds,val_ds,class_weights_dict,use_early_stopping,metric,patience)
    model_stats(fe,history,model,test_ds,class_names[n],thres,prop_speed_pix_frame,num_slices)

    lines_len_hist.append(lines_len)
    history_hist.append(history)
    model_hist.append(model)
    fe_hist.append(fe)      
    be_hist.append(be)         # some of this saving might be unnecessary
    
#%% Save off the model, if desired
#model_hist[0].save('C:\\Users\\tyler\\Desktop\\NSSSIP25\\TrainedModels\\model123.keras')

#%% Read in whole-set classification data 

Imagelist, Imagelist_raw, WP_io, lines_len, img_count, num_slices, height = write_data_whole(whole_set_file_name, slice_width)

#%% Classify the whole-set
print(f'Predicting {class_names[0]}')
confidence_history = model_hist[0].predict(Imagelist)   #classify based on first positive class

#%% Post-process

processed_conf_hist = post_process_time(confidence_history,prop_speed_pix_frame,slice_width,num_slices)
print('Done post-processing (time)')

#%% Visualize results
    
if plot_flag == 1:
    visualize_frames(lines_len,N_frames,num_slices,Imagelist_raw,confidence_history,processed_conf_hist,thres,slice_width)
