# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 12:37:59 2025

@author: cathe
"""
import os
import numpy as np

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
    vmin = np.min(image_line)
    vmax = np.max(image_line)
    # Figure specs
    stretch_y = 2.5;
    aspect_ratio = stretch_y*image_size[0]/image_size[1]

    Imagelist.append(image_line)
    
    if sm_check.startswith('X'):
        WP_io.append(0)
        SM_bounds_Array.append(sm_bounds)

    else:
        WP_io.append(1)
        SM_bounds_Array.append(sm_bounds)
        
#%%

slice_WP =  WP_io[0:125]
        
total_img = len(Imagelist)
total_WP = slice_WP.count(1)

percent_WP = (total_WP/125)*100

print(f"Number of Images in Data Set: {total_img}")
print(f"Percent of images with wave packet: {percent_WP}%")











