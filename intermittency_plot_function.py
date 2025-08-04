
import numpy as np
import matplotlib.pyplot as plt
import copy

def initialize_Langley_vals():
    '''
    Initializes mm_pix, FR, start_mm for Langley runs from https://link.springer.com/article/10.1007/s00348-023-03758-w
    Initializes approximate prop_speed for Langley Re9.7 from above paper

    OUTPUTS
    -------
    mm_pix : float
        mm to pixel conversion for camera set up in above paper
    FR : float
        Camera frame rate in Hz
    prop_speed : int
        A priori estimate of propagation speed, m/s, for Re9.7
    start_mm : int or float
        mm location on cone where sclieren field of view begins

    '''
    mm_pix = 0.0756     
    FR = 285e3          
    prop_speed = 825      
    start_mm = 288
    return mm_pix, FR, prop_speed, start_mm

def intermittency_plot(frame_res_history,start_mm,mm_pix,slice_width):
    """
    Plots an intermittency plot for the detected feature.
    Assumes all frames are the same length.

    INPUTS
    ----------
    frame_res_history : array of arrays for every frame; arrays for each frame have length equal to the number of slices (num_slices)
        list of numpy arrays with post-processed slice classifications for each frame
    
    start_mm: int or float,
        downstream mm location on the object surface where Schlieren image begins
        
    mm_pix: int or float,
        mm/pixel conversation factor for camera/lens set up
        
    slice_width: int,
        width of image slices in pixels

    OUTPUTS:
    -------
    None.
    Plots result.
    
    Reference:
    "Global analysis of nonlinear second-mode development in a Mach-6 boundary layer from high-speed schlieren data"
    pg. 24
    """
    num_slices = len(frame_res_history[0])
    N_frames = len(frame_res_history)

    intermittency = np.zeros(num_slices)
    
    for i_iter in range(N_frames):
        for i, cls_result in enumerate(frame_res_history[i_iter]):
            if cls_result == 1:
                intermittency[i] = intermittency[i] + 1
                
    intermittency = intermittency / N_frames
    
    extent = num_slices*slice_width*mm_pix
    max_extent = start_mm+extent
    mm_width = slice_width*mm_pix
    
    x_values = [] # use the middle of slice
    for x in np.arange(start_mm+mm_width/2.0,max_extent,mm_width):
        x_values.append(x)
    x_values = np.array(x_values)
        
    plt.scatter(x_values, intermittency)
    plt.xlabel('Downstream Location (mm)')
    plt.ylabel('Intermittency')
    plt.title('Intermittency over Object Length')
    
    plt.show()
    return x_values, intermittency
    
def calc_prop_speed_pix(mm_pix,FR,prop_speed):
    """
    Calculate the propogation speed in pixels/frame

    INPUTS
    ----------
    mm_pix: int or float,
        mm/pixel conversation factor for camera/lens set up
        
    FR : float
        Camera frame rate in Hz
        
    prop_speed : int or float
        A priori estimate of propagation speed, m/s

    OUTPUTS
    -------
    PS_pix_frame : float
        propagation speed in pixels/frame

    """
    dt = 1/FR                     # Time step between frames
    PS_pix_frame = prop_speed * dt * 1/(mm_pix*1e-3)  # Computed number of pixels traveled between frames
    return PS_pix_frame
