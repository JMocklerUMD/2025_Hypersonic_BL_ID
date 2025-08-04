# MATLAB Utility Code

Hypersonic boundary layer feature identification using ML-enabled components

## Files

Note: many files contain specific file paths to images or folders that need to be changed and/or added to path before running



* AlternativeMethodsAccuracy: use to compare accuracy of classification methods, input training data (folder of .tif images and correctly formatted text file), outputs accuracy, TPR, TNR, FPR, and FNR
* coneflare\_split: work-in-progress (never implemented and fully tested), splits cone flare images at corner point, input .tif images, outputs x coordinate of corner by taking average intersection of cone surfaces, plots two halves of split image (could also be modified to save these to text file or folder like with final\_createTD)
* file\_fix: use to invert a set of images (some data was originally processed by doing mean - img instead of img - mean)
* filter\_notch: function used for bandpass filtering, no changes required to this file
* final\_createTD: final labeling/image processing script, will do mean subtraction and filtering, followed by manually drawing of bounding boxes for desired feature, outputs text file and raw .tif images corresponding to the images that were classified
* video\_creation: creates animation of flow from .png frames
