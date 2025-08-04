# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 10:05:44 2025

@author: Joseph Mockler
"""

# imports
import random
import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

import os

def Shuffler(list1, list2):
    # Unused, inherited from older code
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
    # Unused, inherited from older code
	percent = ("{0:." + str(decimals) + "f}").format(100*(iteration/float(total)))
	filledLength = int(length*iteration//total)
	bar = fill*filledLength + '-'*(length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
	if iteration == total:
		print()
        
        
def get_bottleneck_features(model, input_imgs, verbose = 0):
    '''
    Extracts the features from a slice using a pre-trained CNN
    
    INPUTS
    ------------------
    model: keras pretrained CNN, 
        the j'th slice of the frame to calculate the confidence
    
    input_imgs: N dim array of 224x224 np arrays, 
        rescaled image slices to be classified by the resnet50 model
    
    verbose: int, default = 0, 
        1 if wanting to print when features are extracted
    
    OUTPUTS
    ------------------
    features: N dim array of 1D np arrays, 
        pretrained CNN features that are extracted from the slices and used to train the top-level classifier
    '''

    if input_imgs.shape == (224,224,3): #adds batch dimension for single images
        input_imgs = np.expand_dims(input_imgs, axis=0)                # Shape: (1, 224, 224, 3)
    if verbose == 1:
        print('Getting Feature Data From ResNet...')
    features = model.predict(input_imgs, verbose = verbose)
    return features

def img_preprocess(input_image):
    '''
    Extracts the features from a slice using a pre-trained CNN
    
    INPUTS
    ------------------
    input_image: (height,slice_width) numpy array, 
        float image data prescaled between [0,1]
    
    OUTPUTS
    ------------------
    processed_image: (224, 224, 3) numpy array,
        stretched input data to feed into pretrained classifier
    '''

    input_image = np.stack((input_image,)*3,axis = -1)
    input_image = array_to_img(input_image)
    input_image = input_image.resize((224,224))
    processed_image = img_to_array(input_image)
    #input_image = (input_image / 127.5) - 1
    return processed_image


def image_splitting(i, lines, slice_width):
    '''
    Extracts the image and bounding box information from the pre-defined text file.
    This text file must be generated in MATLAB first using the correct script.
    
    INPUTS
    ------------------
    i: int, 
        integer defining which image to read off
        
    lines: MATLAB-generated text file data as a list,
        file containing image data and human labeled bounding box information. GENERATE WITH write_text_to_code(file_name)
        
    slice_width: int,
        width of slice to divide the frame into
    
    OUTPUTS
    ------------------
    Imagelist: N array of (height,slice_width) numpy array, 
        Sliced frame of image data as read from the text file
        
    WP_io: N list of 0/1 binary,
        0/1 binary defining if a slice has a WP or not per pre-defined bounding boxes
    
    slice_width: int,
        slice_width, again. Not quite sure why we still return it, but I don't want to fix a bunch of code
    
    height: int,
        image height in pixels
    
    sm_bounds: 4x1 list of ints,
        defines the human-labeled bounding box (upper left corner x, UL corner y, width, height)    
    '''
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


def write_text_to_code(file_name):
    '''
    Extracts the image and bounding box information from the pre-defined text file.
    This text file must be generated in MATLAB first using the correct script.
    
    INPUTS
    ------------------
    file_name: str, 
        File path to the MATLAB-processed image data text file
        
    OUTPUTS
    ------------------
    lines: MATLAB-generated text file,
        file containing image data and human labeled bounding box information
        
    lines_len: int,
        number of labeled images read in
    '''
    
    print("Begin reading the MATLAB labeled image text file")
    # Write File Name
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()
    else:
        raise ValueError("No training_data file detected")

    lines_len = len(lines)
    print(f"Completed! {lines_len} lines read")
    
    return lines, lines_len


def resize_frames_for_ResNet50(Imagelist, WP_io):
    '''
    Extracts the image and bounding box information from the pre-defined text file.
    This text file must be generated in MATLAB first using the correct script.
    
    INPUTS
    ------------------
    Imagelist: N array of (height, slice_width) arrays, 
        a list of list that each contain an sliced image
    
    WP_io: N list of 0/1 ground truth binaries
        a corresponding list that classifies each sliced image as 1 or 0, int
        
    OUTPUTS
    ------------------
    Imagelist_resized:  N array of (224x224) arrays
        frame resized to 224x224 (resnet50 required input). This is performed by img_preprocess
    '''
    
    # Resizes the arrays
    Imagelist = np.array(Imagelist)
    WP_io = np.array(WP_io)
    Imagelist_resized = np.array([img_preprocess(img) for img in Imagelist])
    print("Done Resizing")
    
    return Imagelist_resized