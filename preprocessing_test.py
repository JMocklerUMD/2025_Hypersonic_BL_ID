# -*- coding: utf-8 -*-
"""
Image Preprocessing Script

@author: rclat
"""

#%% Import libraries
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import exposure
from PIL import Image, ImageDraw

#%% Read in image(s)
image = tiff.imread('C:\\Users\\rclat\\Downloads\\img58.tiff')

plt.imshow(image,cmap='gray')
plt.title('TIFF Image')
plt.axis('off')
plt.show()


#%% Example using FFT in Python
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

#%%
magnitude_spectrum = np.abs(fshift)

# Apply logarithmic scaling to improve visibility of small features
log_magnitude = np.log1p(magnitude_spectrum)  # same as log(1 + |F|)

rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
radius_inner = 20  # adjust as needed
radius_outer = 45  # adjust as needed

# Create circular bandpass mask
Y, X = np.ogrid[:rows, :cols]
dist_from_center = np.sqrt((X - ccol)**2 + (Y - crow)**2)
mask = (dist_from_center >= radius_inner) & (dist_from_center <= radius_outer)

# ---- Plot log-scaled magnitude with mask overlay ----
plt.figure(figsize=(8, 8))
plt.imshow(log_magnitude, cmap='gray')
plt.title('Log-Scaled FFT Magnitude with Bandpass Overlay')
plt.axis('off')

# Overlay the mask boundary (use a red contour)
plt.contour(mask.astype(int), levels=[0.5], colors='red', linewidths=0.8)

plt.show()

#%% Design a bandpass mask to isolate second-mode spatial frequencies
rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2
mask = np.zeros((rows, cols), dtype=np.uint8)
radius_inner = 20
radius_outer = 45
for i in range(rows):
    for j in range(cols):
        r = np.sqrt((i - crow)**2 + (j - ccol)**2)
        if radius_inner < r < radius_outer:
            mask[i,j] = 1

fshift_filtered = fshift * mask
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.abs(np.fft.ifft2(f_ishift))

plt.imshow(img_back)
plt.title('filtered TIFF Image (circular filter)')
plt.axis('off')
plt.show()

#%%
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Frequency range for horizontal spike filtering (adjust these)
x_low = 20   # left/right from center
x_high = 45

# Create directional mask: band along x-axis
mask = np.zeros((rows, cols), dtype=np.uint8)
for i in range(rows):
    for j in range(cols):
        dx = abs(j - ccol)
        dy = abs(i - crow)
        if x_low < dx < x_high and dy < 10:  # limit vertical range too
            mask[i, j] = 1
            
fshift_filtered = fshift * mask
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.abs(np.fft.ifft2(f_ishift))

plt.imshow(img_back)
plt.title('filtered TIFF Image (rectangular filter)')
plt.axis('off')
plt.show()


#%% Changing exposure (seems to make wave packets harder to see)

'''
img_back = img_back.astype(np.float32)
image_norm = (img_back - img_back.min()) / (img_back.max() - img_back.min())
image_eq = exposure.equalize_adapthist(image_norm, clip_limit=0.03)

plt.imshow(image_eq)
plt.title('exposure TIFF Image')
plt.axis('off')
plt.show()
'''
