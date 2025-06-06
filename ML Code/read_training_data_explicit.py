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
file_name = 'C:\\Users\\Ryan de Silva\\Desktop\\UMD Research\\Machine Learning\\Wave-Packet Identification\\training_data_explicit.txt'
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
    image = np.array(image_data).astype(np.float64)
    image = image.reshape(image_size)  # Reshape to (rows, columns)

    print(f"Number of Parts: {len(parts)}")
    print(f"Run: {run}")
    print(f"SM Bounds: {sm_bounds}")
    print(f"Image Size: {image_size}")


    vmin = np.min(image)
    vmax = np.max(image)
    # Figure specs
    stretch_y = 2.5;
    aspect_ratio = stretch_y*image_size[0]/image_size[1]
    figsize = (20,20*aspect_ratio)
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    # Display image
    ax.imshow(image, cmap='bone', vmin=vmin, vmax=vmax)
    Imagelist.append(image)
    
    if sm_check.startswith('X'):
        WP_io.append(0)
        plt.show()
        print('No wave-packet detected')
    else:
        WP_io.append(1)
        SM_bounds_Array.append(sm_bounds)
        rect = plt.Rectangle((sm_bounds[0], sm_bounds[1]), sm_bounds[2], sm_bounds[3], edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        fig.canvas.draw()
        plt.show()

Imagelist,WP_io,SM_bounds_Array = Shuffler(Imagelist, WP_io, SM_bounds_Array)

Imagelist = np.array(Imagelist)
WP_io = np.array(WP_io)
SM_bounds_Array = np.array(SM_bounds_Array)


