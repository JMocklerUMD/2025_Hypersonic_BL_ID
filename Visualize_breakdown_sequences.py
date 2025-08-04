# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:16:42 2025

@author: tyler
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from turbulence_and_breakdown_utils import *
from ML_utils import write_text_to_code
from Results_utils import whole_image

#   -   -   -   -   EXPERIMENTIAL CODE   -   -   -   -   EXPERIMENTIAL CODE   -   -   -   -   EXPERIMENTIAL CODE   -   -   -   -

slice_width = 64
Langley = True
file_name = "C:\\Users\\tyler\\Desktop\\NSSSIP25\\ClassificationAndImageData\\run33CloserBreakdownCropped\\video_data.txt"

turb_result_history = np.load('C:\\Users\\tyler\\Desktop\\NSSSIP25\\ClassificationAndImageData\\run33CloserBreakdownCropped\\turb_result_history.npy')
wp_result_history = np.load('C:\\Users\\tyler\\Desktop\\NSSSIP25\\ClassificationAndImageData\\run33CloserBreakdownCropped\\wp_result_history.npy')

if Langley:
    mm_pix, FR, prop_speed, start_mm = initialize_Langley_vals() 
    # these parameters will have to be defined if not using Langley images
    '''
    mm_pix : float
        mm to pixel conversion for camera set up in above paper
    FR : float
        Camera frame rate in Hz
    prop_speed : int
        A priori estimate of propagation speed, m/s, for Re9.7
    start_mm : int or float
        mm location on cone where sclieren field of view begins
    '''
PS_pix_frame = calc_prop_speed_pix(mm_pix,FR,prop_speed)
    
# generate stats and find sequences of WP breakdown or upstream turbulence
breakdown_sequences, upstream_turb_sequences = breakdown_stats(wp_result_history,turb_result_history,slice_width,PS_pix_frame)

lines, lines_len = write_text_to_code(file_name)

#%%
gap = 2 # show every gap images (ex: 2 would show every other images while 1 shows every single image)
len_thres = 12 # minimum length of sequences to visualize

for N,[end,start] in enumerate(upstream_turb_sequences): #change this input to show either sequences of WP breakdown or upstream turbulence
    if end-start+1 > len_thres: #9 seemed to be the lowest value at the current set up
        fig, axs = plt.subplots(len(range(start,end+1,gap)))
        for n in range(start,end+1,gap):
            fullImage = whole_image(n, lines)
            axs[(n-start)//gap].imshow(fullImage,cmap='grey')
            axs[(n-start)//gap].axis('off')
            
            # Add on classification box rectangles
            for i,result in enumerate(wp_result_history[n]):    
                # Add in the WP classification guess
                if result == 1:
                    rect = Rectangle((i*slice_width, 5), slice_width, slice_width-10,
                                             linewidth=0.5, edgecolor='red', facecolor='none')
                    axs[(n-start)//gap].add_patch(rect)
            for i,result_turb in enumerate(turb_result_history[n]):    
                # Add in the turbulence classification guess
                if result_turb == 1:
                    rect = Rectangle((i*slice_width, 5), slice_width, slice_width-10,
                                             linewidth=0.5, edgecolor='yellow', facecolor='none')
                    axs[(n-start)//gap].add_patch(rect)
            plt.title(f'Frames {start} to {end}')
                
        plt.show()
        
#%% save off some .mat files for plotting in Matlab
from scipy.io import savemat
import numpy as np

for number in np.arange(1309,1322,2):
    fullImage = whole_image(number, lines)
    mdic = {f'frame{number}': fullImage}
    name = f'frame{number}.mat' 
    savemat(name,mdic)