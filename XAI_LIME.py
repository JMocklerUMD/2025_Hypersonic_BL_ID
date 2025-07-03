# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:01:26 2025

@author: cathe
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import os

import keras
import tensorflow as tf

from keras.applications import resnet50
from keras.models import Model

from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from skimage import img_as_ubyte

explainer = lime_image.LimeImageExplainer()


#%% Function Calls + Resnet50 instantiation
# This function tells our feature extractor to do its thing
def get_bottleneck_features(model, input_imgs):
	features = model.predict(input_imgs, verbose = 0)
	return features

def img_preprocess(input_image):
    if input_image.ndim == 2:
        input_image = np.stack((input_image,) * 3, axis=-1)
    input_image = array_to_img(input_image)
    input_image = input_image.resize((224,224))
    input_image = img_to_array(input_image)
    #input_image = (input_image / 127.5) - 1
    return input_image

def lime_model_predict(images):
    images_preprocessed = np.array([img_preprocess(img) for img in images])
    features = get_bottleneck_features(resnet_model, images_preprocessed)
    probs = model.predict(features, verbose=0)
    return np.concatenate([(1 - probs), probs], axis=1)

def get_lime_overlay(image_slice, label, num_features=5, num_samples=100):
    img_rgb = img_preprocess(image_slice) / 255.0
    explanation = explainer.explain_instance(
        img_rgb.astype(np.double),
        lambda imgs: lime_model_predict(imgs),  # defined below
        labels=[label],
        hide_color=0,
        num_samples=num_samples
    )
    temp, mask = explanation.get_image_and_mask(
        label=label,
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )
    color = (0, 1, 0) if label == 1 else (1, 0, 0)
    overlay = mark_boundaries(temp, mask, color=color, mode='thick')
    return overlay


model1 = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
output = model1.output
output = tf.keras.layers.Flatten()(output)
resnet_model = Model(model1.input,output)

# load the classifier
model = keras.models.load_model("C:\\Users\\cathe\\OneDrive\\Desktop\\WPML\\T9Model_Normalized.keras")


#%% read in images
print('Reading training data file')

# Write File Name
file_name ="C:\\Users\\cathe\\OneDrive\\Desktop\\T9_Run4120_Normalized.txt"
if os.path.exists(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len = len(lines)
print(f"{lines_len} lines read")


#%% Split the image into 20 pieces
def image_splitting(i, lines):
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

    slice_width = 54
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

def classify_the_images(model, Imagelist):
    Imagelist_resized = np.array([img_preprocess(img) for img in Imagelist])
    
    # Run through feature extractor
    Imagelist_res = get_bottleneck_features(resnet_model, Imagelist_resized)
    
    # Pass each through the trained NN
    test_res= model.predict(Imagelist_res)
    classification_result = np.round(test_res)
    
    return classification_result, test_res


#%% Iterate through the list!
N_img = lines_len
plot_flag = 1    # View the images? MUCH SLOWER

for i_iter in range(10):
    
    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines)
    
    classification_result, confidence = classify_the_images(model, Imagelist)
    
    if plot_flag == 1:
        # --- LIME overlays for each slice ---
        lime_overlays_resized = []
        for i, slice_img in enumerate(Imagelist):
            print(f'Slice {i+1}/10')
            pred_label = int(np.round(confidence[i][0]))
            overlay_img = get_lime_overlay(slice_img, label=pred_label, num_features=5, num_samples=100)
            resized = resize(overlay_img, (height, slice_width, 3), preserve_range=True)
            resized = img_as_ubyte(resized)
            lime_overlays_resized.append(resized)
    
        stitched_overlay = np.hstack(lime_overlays_resized)
    
        # --- Plot with LIME overlays and proper axis style ---
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.imshow(stitched_overlay)
    
        # Tick locations at slice centers
        ax.set_xticks([j * slice_width + slice_width // 2 for j in range(len(confidence))])
        ax.set_xticklabels([f"{confidence[j][0]:.2f}" for j in range(len(confidence))], fontsize=7)
    
        ax.set_yticks([0, height])  # just top and bottom
        ax.set_ylabel("0\n\n" + str(height),va='center')
        ax.set_xlabel("Green = class 1, Red = class 0")
    
        # Red rectangles for class=1
        for j in range(len(classification_result)):
            if classification_result[j] == 1:
                rect = Rectangle((j*slice_width, 5), slice_width, height-10, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
    
        # Ground truth bounding box in blue
        if sm_bounds[0] != 'X':
            ax.add_patch(Rectangle((sm_bounds[0], sm_bounds[1]), sm_bounds[2], sm_bounds[3],
                                   edgecolor='blue', facecolor='none', linewidth = 2))
    
        ax.set_title(f"Image {i_iter}. Blue: true WP. Red: NN class")
        ax.set_ylim(height, 0)  # invert y-axis to match grayscale style
        plt.tight_layout()
        plt.show()

