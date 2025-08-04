# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 10:08:14 2025

@author: Joseph Mockler
"""

import numpy as np
from ML_utils import get_bottleneck_features, img_preprocess
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def calc_windowed_confid(j, confidence, window_size):
    '''
    Calculates the local confidence (i.e. a single slice of a frame) 
    via a summed windowing method
    
    INPUTS
    ------------------
    j: int, 
        the j'th slice of the frame to calculate the confidence
    
    confidence: 1D array, 
        array of [0,1] classification confidences corresponding to the slices in a frame
    
    window_size: int, 
        the window (centered on j) to search over. MUST BE ODD
    
    OUTPUTS
    ------------------
    local_confid: float, 
        the windowed summed confidence of slice j
    '''
    
    if (j - window_size//2) < 0: # at the front end of the image
        local_confid = np.sum(confidence[0:j+window_size//2+1:1])
    elif (j + window_size//2) > len(confidence): # at the end of the image list
        local_confid = np.sum(confidence[j-window_size//2-1:len(confidence):1])
    else:
        local_confid = np.sum(confidence[j-window_size//2:j+window_size//2+1:1])
        
    return local_confid

def classify_WP_frame(Imagelist, WP_io, confidence, window_size, indiv_thres, confid_thres, use_post_process):
    """
    Classifies all slices in an entire frame via either the post-processing window
    method or via simple threshold rounding.
    
    INPUTS
    ------------------
    Imagelist: list of 2D arrays, 
        list of slices that compose an entire frame
    
    WP_io: list, 0 or 1 binary, 
        the ground truth corresponding exactly to the slices in Imagelist. If no ground truth, pass zeros
    
    confidence: 1D array, 
        array of [0,1] classification confidences corresponding to the slices in a Imagelist
    
    window_size: int, 
        the window (centered on j) to search over. MUST BE ODD
    
    indiv_thres: float, 
        breakout threshold value for individual exceptions outside of windowing
    
    confid_thres: float, 
        threshold for windowing. NOTE: use the SUMMED value (e.g. 0.5 per window * 3 windows = 1.5 confid_thres)
    
    use_post_process: int, 
        0 or 1, 1 for windowing, 0 for simple rounding
                
    OUTPUTS
    ------------------
    classification_results: list, 0 or 1's, 
        the filtered or rounded results corresponding to Imagelist slices
    
    n11, n10, n01, n00: ints, 
        confusion statistics corresponding to the entire frame. n11 = TP, n00 = TN, n10 = FN, n01 = FP
    """
    
    filtered_result = []
    classification_result = np.zeros(np.array(WP_io).shape)
    for i, _ in enumerate(confidence):
        # If using the windowed post processing, call the windowing fcn
        # to get the locally informed confidence. Then compare to thresholds
        if use_post_process == 1:
            local_confid = calc_windowed_confid(i, confidence, window_size)
            
            # Are window and indiv conditions met?
            if (local_confid > confid_thres) or (confidence[i] > indiv_thres):
                filtered_result.append(1)
            else:
                filtered_result.append(0)
            
            classification_result[i] = filtered_result[i]
        
        # If not, then just round
        else:
            classification_result[i] = np.round(confidence[i])

    # Now get the confusion statistics from the frame  
    n11, n10, n01, n00 = confusion_results(classification_result, WP_io)
        
    # and return the frame-based statistics and the finalized results
    
    return classification_result, n11, n10, n01, n00


def form_ROC_PR_statistics(Imagelist, WP_io_history, confidence_history, window_size, use_post_process):
    """
    Forms the ROC and PR curve data for a classifier using the history of [0,1] classification
    values for each frame. Returns a 50x1 array of data needed for an ROC or PR curve
    
    INPUTS
    ------------------
    Imagelist: list of 2D arrays, 
        list of slices that compose an entire frame
    
    WP_io_history: list of lists, inner lists of 0/ binary, 
        the ground truth corresponding exactly to the slices in Imagelist. Inner list is a single frame, outer list is all frames in a labeled video
    
    WP_io_history: list of 1D arrays, 
        array of [0,1] classification confidences corresponding to the slices in a Imagelist. Inner list is a single frame, outer list is all frames in a classified video
    
    window_size: int, 
        the window (centered on j) to search over. MUST BE ODD
        
    use_post_process: int, 
        0 or 1, 1 for windowing, 0 for simple rounding
                
    OUTPUTS
    ------------------
    FPRs: list (50x1),
        Entire video false positive rate for an increasing [0,1] class. threshold
    
    TPRs: list (50x1),
        Entire video true positive rate for an increasing [0,1] class. threshold
        
    Pres: list (50x1),
        Entire video precision for an increasing [0,1] class. threshold
    """
    
    # Form an ROC curve
    if use_post_process:
        thresholds = np.linspace(0, window_size, num=50)
    else:
        thresholds = np.linspace(0, 1, num = 50)
        
    TPRs, FPRs, Pres = [], [], []
    # Loop thru the thresholds
    for threshold in thresholds:
        TP, FP, TN, FN = 0, 0, 0, 0
        
        # Loop thru each image in the test set
        for i in range(len(WP_io_history)):

            # Pull off the sliced list
            WP_io = WP_io_history[i]
            confidence = confidence_history[i]
            
            # Classify the frame via post processing or simple threshold based on
            # the use_post_process flag (NOTE, unsure if this works for turb too)
            classification_result, n11, n10, n01, n00 = classify_WP_frame(Imagelist, WP_io, confidence, window_size, threshold/window_size, threshold, use_post_process)
            
            # Finally, add to the grand list per threshold
            TP = TP + n11
            FP = FP + n01
            TN = TN + n00
            FN = FN + n10
            
        # Now calculate the percentages
        TPRs.append(TP/(TP+FN))
        FPRs.append(FP/(FP+TN))
        if (TP+FP) == 0:
            Pres.append(1.0)
        else:
            Pres.append(TP/(TP+FP))
            
    return FPRs, TPRs, Pres

def form_ROC_Curve(Imagelist, WP_io_history, confidence_history, window_size, use_post_process):
    """
    Plots the ROC and PR curves for a classifier and experiment.
    
    INPUTS
    ------------------
    Imagelist: list of 2D arrays, 
        list of slices that compose an entire frame
    
    WP_io_history: list of lists, inner lists of 0/ binary, 
        the ground truth corresponding exactly to the slices in Imagelist. Inner list is a single frame, outer list is all frames in a labeled video
    
    WP_io_history: list of 1D arrays, 
        array of [0,1] classification confidences corresponding to the slices in a Imagelist. Inner list is a single frame, outer list is all frames in a classified video
    
    window_size: int, 
        the window (centered on j) to search over. MUST BE ODD
    
    use_post_process: int, 
        0 or 1, 1 for windowing, 0 for simple rounding
                
    OUTPUTS
    ------------------
    AUC: float,
        Area under the ROC curve
        
    PR: float,
        Area under the PR curve
        
    fig, ax, ax2: matplotlib.pyplot objects,
        Figure handles for the ROC and PR curves that are plotted to the console
    """
    
    # Extract FPs, TPs, Pres results
    FPRs, TPRs, Pres = form_ROC_PR_statistics(Imagelist, WP_io_history, confidence_history, window_size, use_post_process)
    
    # Compute the AUC of the ROC - simple rectangular integration
    AUC = 0.0
    for i in range(1,len(TPRs)):
        AUC = AUC + (FPRs[i-1]-FPRs[i])*(TPRs[i]+TPRs[i-1])/2    
    print(f'Area under the ROC Curve = {AUC}')
    
    PR = 0.0
    for i in range(1,len(TPRs)):
        PR = PR + (Pres[i]+Pres[i-1])*(TPRs[i-1]-TPRs[i])/2    
    print(f'Area under the PR Curve = {PR}')
    
    # Plot the curve
    fig, (ax, ax2) = plt.subplots(1,2, figsize = (16,8))
    ax.plot(FPRs, TPRs, '--.', markersize=10)
    ax.plot(np.linspace(0,1,num=100), np.linspace(0,1,num=100))
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    ax2.plot(TPRs, Pres, '--.', markersize=10)
    ax2.plot(np.linspace(0,1,num=100), np.flip(np.linspace(0,1,num=100)))
    ax2.set_title('Precision-Recall Curve')
    ax2.set_xlabel('Recall (True Positive Rate)')
    ax2.set_ylabel('Precision')
    
    plt.show()
    
    return AUC, PR, fig, ax, ax2

def plot_frame(FrameList, frame_res, confidence, slice_width, second_mode, turb):
    """
    Plots a single Schlerien image frame with NN classification results (per slice) overlaid.
    Functionality does support turbulence labeling, but needs additional development to confirm.
    
    INPUTS
    ------------------
    FrameList: list of 2D arrays, 
        list of slices that compose an entire frame
        
    frame_res: list of 0/1 binarys,
        list of classification results corresponding to each frame. 0 is no label, 1 is true
        
    confidence: 1D array, 
        array of [0,1] classification confidences corresponding to the slices in a FrameList
    
    slice_width: int,
        pixel width of a single slice in a frame.
        
    second_mode: 0, 1 binary
        1 if we're classifying a 2nd mode WP, 0 otherwise
        
    turb: 0, 1 binary,
        1 if we're classifying turbulence, 0 otherwise        
                
    OUTPUTS
    ------------------  
    fig, ax: matplotlib.pyplot objects,
        Figure handles for the ROC and PR curves that are plotted to the console
    """
    
    imageReconstruct = np.hstack([image for image in FrameList])
    fig, ax = plt.subplots(1)
    ax.imshow(imageReconstruct, cmap = 'gray')
    
    height, width = imageReconstruct.shape
    
    if second_mode:
        ax.text(-45, height+86,'WP: ', fontsize = 6)
    if turb:
        ax.text(-57, height+62,'Turb: ', fontsize = 6)
    
    # Add on classification box rectangles
    for i, _ in enumerate(FrameList):    
        # Add in the classification guess
        if frame_res[i] == 1:
            rect = Rectangle((i*slice_width, 5), slice_width, height-10,
                                     linewidth=0.5, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
        prob = round(confidence[i,0],2)
        ax.text(i*slice_width+slice_width/5, height+86,f'{prob:.2f}', fontsize = 6)
        
    return fig, ax

def moving_average(a, n):
    '''
    Moving average of a list, with a being a list of floats and n being the moving average window.
    '''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def whole_image(i, lines):
    """
    Returns an entire image from the written text file repository
    
    INPUTS
    ------------------
    i: int, 
        i'th frame of lines to extract the image from
        
    lines: text file,      
        MATLAB-generated repository of video data read from the text file          
      
    OUTPUTS
    ------------------  
    full_img: 2D np array,
        MxN array of an entire frame.
    """
    curr_line = i;
    line = lines[curr_line]
    
    parts = line.strip().split()
    
    image_size = list(map(int, parts[6:8]))  # Convert image size to integers
    image_data = list(map(float, parts[8:]))  # Convert image data to floats
    
    # Reshape the image data into the specified image size
    full_image = np.array(image_data).astype(np.float64)
    full_image = full_image.reshape(image_size)  # Reshape to (rows, columns)
    
    return full_image

def classify_the_images(model, resnet_model, Imagelist, verbose=0): #not verbose by default
    """
    Classifies the slices of a frame using a transfer learning (custom NN) approach.
    May handle either WP classification or turb with an appropriate classification model passed in
    
    INPUTS
    ------------------
    model: keras NN object, 
        top-level NN object that accepts the ResNet50 flattened output from a single square slice and maps to a [0,1]
        
    resnet_model: keras pretrained CNN object,      
        pretrained keras model to extract feature information from the slices
        
    Imagelist: list of 2D arrays,
        A sliced frame (recommended, 64 pixel-width), where each slice will be classified through the transfer learning model
      
    OUTPUTS
    ------------------  
    classification_results: 1D np array,
        the rounded, binary results labeling the slice as containing or not containing a WP
        
    test_res, 1D np array,
        the raw [0,1] classification results mapped per slice
        
    Imagelist_res, list of 1D np arrays
        the resnet_model output results from a slice passed in. Returns list of a feature vector for all slices passed in
    """
    Imagelist_resized = np.array([img_preprocess(img) for img in Imagelist])
    # Run through feature extractor
    Imagelist_res = get_bottleneck_features(resnet_model, Imagelist_resized, verbose)
    
    # Pass each through the trained NN
    test_res= model.predict(Imagelist_res, verbose = 0)
    classification_result = np.round(test_res)
    return classification_result, test_res, Imagelist_res


def confusion_results(prediction, truth):
    """
    On a frame-level, computes the confusion statistics between the NN prediction (either processed or un) and the ground truth
    
    INPUTS
    ------------------
    prediction: array, 
        0/1 binary classifier prediction with length aligned with the number of slices in a frame
        
    truth: array, 
        0/1 binary ground truth (e.g human labeled) prediction with length aligned with the number of slices in a frame
        
    OUTPUTS
    ------------------  
    n11, n10, n01, n00: ints, 
        confusion statistics corresponding to the entire frame. n11 = TP, n00 = TN, n10 = FN, n01 = FP
    """
    n00, n01, n10, n11 = 0, 0, 0, 0 
    
    for i, label_true in enumerate(truth):
        label_pred = prediction[i]
        
        if label_true == 0:
            if label_pred == 0:
                n00 += 1
            if label_pred == 1:
                n01 += 1 
        elif label_true == 1:
            if label_pred == 0:
                n10 += 1
            if label_pred == 1:
                n11 += 1
    
    return n11, n10, n01, n00

class ClassificationStatistics:
    """
    Using confusion statistics for a single frame OR entire video, generates an object computing and saving more advanced statistics
    
    INPUTS
    ------------------
    n11, n10, n01, n00: ints, 
        confusion statistics corresponding to the entire frame. n11 = TP, n00 = TN, n10 = FN, n01 = FP
        
    nsamples: int,
        Number of slices in the frame being classified
        
    ATTRIBUTES
    ------------------  
    TP: True positive count
    TN: True negative count
    FP: False positive count
    FN: False negative count
    acc: accuracy amongst the frame
    
    Se: Sensitivity (advanced statistic, recommend on entire video)
    Recall: Recall (advanced statistic, recommend on entire video)
    Pp: Positive predicitive power (advanced statistic, recommend on entire video)
    Sp: Specificity (advanced statistic, recommend on entire video)
    Np: Negative predicitive power (advanced statistic, recommend on entire video)
    FRP: False positive rate (advanced statistic, recommend on entire video)
    MICE: MICE score (advanced statistic, recommend on entire video)
    
    METHODS
    ------------------
    compute_whole_set_statistics():
        Returns advanced statistics on the frame. NOTE: with no positive class (either ground truth or NN class), will return div by 0 error
        
    print_stats():
        Prints the advanced statistics to console
    """
    
    def __init__(self, n11, n10, n01, n00, nsamples):
        ''' Initialize the basic statistics binary classification info '''
        
        self.n11 = n11
        self.n10 = n10
        self.n01 = n01
        self.n00 = n00
        
        self.N = nsamples
        
        # Compute the aggregate labels
        self.n0 = n00 + n01
        self.n1 = n10 + n11
        
        # Save off more descriptive labels
        self.TP = n11
        self.TN = n00
        self.FP = n01
        self.FN = n10
        
        # Finally, save the frame accuracy
        self.acc = (self.n00 + self.n11) / self.N # complete accuracy
        
    def compute_whole_set_statistics(self):
        # Only call this when you have an entire set of n00, n01, ... to compute
        # otherwise, there's a decent chance you'll end up with a zero division error!
        
        ''' Compute more advanced statistics
        
        # Compute accuracy, sensitivity, specificity, 
        # positive prec, and neg prec
        # As defined in:
            # Introducing Image Classification Efficacies, Shao et al 2021
            # or https://arxiv.org/html/2406.05068v1
            # or https://neptune.ai/blog/evaluation-metrics-binary-classification
            
        '''

        self.Se = self.n11 / self.n1 # true positive success rate, recall
        self.Recall = self.n11/(self.n11+self.n10) # Probability of detection
        self.Pp = self.n11 / (self.n11 + self.n01) # correct positive cases over all pred positive
        self.Sp = self.n00 / self.n0 # true negative success rate
        self.Np = self.n00 / (self.n00 + self.n10) # correct negative cases over all pred negative
        self.FRP = self.n01/(self.n01+self.n00) # False positive, probability of a false alarm

        # Rate comapared to guessing
        # MICE -> 1: perfect classification. -> 0: just guessing
        self.A0 = (self.n0/self.N)**2 + (self.n1/self.N)**2
        self.MICE = (self.acc - self.A0)/(1-self.A0)   
        
    def print_stats(self):
        # Only call this when you have an entire set of n00, n01, ... to compute
        # otherwise, there's a decent chance you'll end up with a zero division error!
        
        self.compute_whole_set_statistics()
        
        print("---------------Test Results---------------")
        print("            Predicted Class         ")
        print("True Class     0        1    Totals ")
        print(f"     0        {self.n00}       {self.n01}    {self.n0}")
        print(f"     1        {self.n10}        {self.n11}    {self.n1}")
        print("")
        print("            Predicted Class         ")
        print("True Class     0        1    Totals ")
        print(f"     0        {self.n00/self.N}      {self.n01/self.N}    {self.n0}")
        print(f"     1        {self.n10/self.N}      {self.n11/self.N}    {self.n1}")
        print("")
        print(f"Model Accuracy: {self.acc}, Sensitivity: {self.Se}, Specificity: {self.Sp}")
        print(f"Precision: {self.Pp},  Recall: {self.Recall}, False Pos Rate: {self.FRP}")
        print(f"MICE (0->guessing, 1->perfect classification): {self.MICE}")
        print("")
        print(f"True Pos: {self.n11}, True Neg: {self.n00}, False Pos: {self.n01}, False Neg: {self.n10}")
        
        