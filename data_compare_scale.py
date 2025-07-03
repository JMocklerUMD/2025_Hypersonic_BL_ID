# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:58:24 2025

@author: cathe
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

print('Reading training data file')

# Write File Name
file_name ="C:\\Users\\cathe\\OneDrive\\Desktop\\T9_Run4120_normalized.txt"
if os.path.exists(file_name):
    with open(file_name, 'r') as file:
        lines_og = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len_og = len(lines_og)
print(f"{lines_len_og} lines read")

#%%

print('Reading training data file')

# Write File Name
file_name ="C:\\Users\\cathe\\OneDrive\\Desktop\\CF_Re45_normalized.txt"
if os.path.exists(file_name):
    with open(file_name, 'r') as file:
        lines_stretched = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len_stretched = len(lines_stretched)
print(f"{lines_len_stretched} lines read")


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
    
    #slice_width = 96
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


#%%

# Run image_splitting on the first line
Imagelist_og, WP_io, slice_width, height, sm_bounds = image_splitting(0, lines_og, 54)

Imagelist_stretched, WP_io, slice_width, height, sm_bounds = image_splitting(0, lines_stretched, 54)


# Plot the first 5 split images
for idx in range(20):
    img_og  = Imagelist_og[idx]
    img_og_resized = resize(img_og, (224,224))
    plt.figure()
    plt.imshow(img_og_resized, cmap='gray')
    plt.title('T9')
    plt.axis('off')
    plt.show()
    
    img_stretched  = Imagelist_stretched[idx]
    img_stretched_resized = resize(img_stretched, (224,224))
    plt.figure()
    plt.imshow(img_stretched_resized, cmap='gray')
    plt.title("Cone FLare")
    plt.axis('off')
    plt.show()
    
    diff = np.abs(img_og_resized - img_stretched_resized)

    # Plot difference
    plt.figure()
    plt.imshow(diff, cmap='hot')  # Use a color map to enhance visibility
    plt.title("Absolute Difference")
    plt.axis('off')
    plt.colorbar(label='Pixel Difference')
    plt.show()
    
    