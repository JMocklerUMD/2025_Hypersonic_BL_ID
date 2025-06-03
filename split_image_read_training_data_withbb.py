# import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from PIL import Image, ImageChops

import numpy as np
import os
import math
import random

#%% Functions

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
SM_bounds_Array = []
Imagelist = []

for i in range(50):
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
            plt.imshow(image, cmap='bone')
            plt.axis('off')
            plt.show()
            print(f"SM Bounds: {sm_bounds}")
            print("No wave-packet detected")
            SM_bounds_Array.append(sm_check)
        else:
            # Check for horizontal overlap with this slice
            if x_max > x_start and x_min < x_end:
                WP_io.append(1)
    
                # Adjust bounding box coordinates to this slice
                adj_x_min = max(0, x_min - x_start)
                adj_x_max = min(slice_width, x_max - x_start)
                adj_width = adj_x_max - adj_x_min
                adj_bbox = [adj_x_min, y_min, adj_width, box_height]
                SM_bounds_Array.append(adj_bbox)
                print(f"SM Bounds: {adj_bbox}")
    
                # Visualization
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(image, cmap='bone')
                ax.add_patch(plt.Rectangle((adj_x_min, y_min), adj_width, box_height,
                                           edgecolor='red', facecolor='none', linewidth=2))
                ax.axis('off')
                plt.show()
                print('wave-packet detected')
            else:
                WP_io.append(0)
                SM_bounds_Array.append(['X', 'X', 'X', 'X'])
                print('No wave-packet detected')
                
        print(f"Run: {run}")
        print(f"Image Size: {image_size}")
        
        
   
Imagelist,WP_io,SM_bounds_Array = Shuffler(Imagelist, WP_io, SM_bounds_Array)

Imagelist = np.array(Imagelist)
WP_io = np.array(WP_io)
SM_bounds_Array = np.array(SM_bounds_Array)