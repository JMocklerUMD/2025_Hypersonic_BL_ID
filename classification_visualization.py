import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import os

import keras
import tensorflow as tf

from keras.applications import resnet50
from keras.models import Model

#from keras.preprocessing import image
#from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img


#%% Function Calls + Resnet50 instantiation
# This function tells our feature extractor to do its thing
def get_bottleneck_features(model, input_imgs):
	print('Getting Feature Data From ResNet...')
	features = model.predict(input_imgs, verbose = 0)
	return features

def img_preprocess(input_image):
    input_image = np.stack((input_image,)*3,axis = -1)
    input_image = array_to_img(input_image)
    input_image = input_image.resize((224,224))
    input_image = img_to_array(input_image)
    #input_image = (input_image / 127.5) - 1
    return input_image


model1 = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
output = model1.output
output = tf.keras.layers.Flatten()(output)
resnet_model = Model(model1.input,output)

# load the classifier
model = keras.models.load_model('C:\\Users\\tyler\\Desktop\\NSSSIP25\\TrainedModels\\Langley_200imgs_with32nodes.keras')


#%% read in images
print('Reading training data file')

# Write File Name
file_name = "C:\\Users\\tyler\\Desktop\\NSSSIP25\\merged_Langley_runs.txt"
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
    
    slice_width = 64
    height, width = full_image.shape
    num_slices = width // slice_width
    
    #chops off black box for longer uncropped images in run34 Langley
    if full_image.shape == (64,1280):
        num_slices = num_slices-1
    
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
    test_res= model.predict(Imagelist_res, verbose = 0)
    classification_result = np.round(test_res)
    
    return classification_result, test_res


#%% Iterate through the list!
N_img = lines_len
acc_history = []
TP_history = []
TN_history = []
FP_history = []
FN_history = []
WP_io_history = []
confidence_history = []
plot_flag = 1       # View the images? MUCH SLOWER

for i_iter in range(N_img):
    plot_flag = 0  #i_iter%10
    
    
    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines)
    
    classification_result, confidence = classify_the_images(model, Imagelist)
  
    # Restack and plot the image
    imageReconstruct = np.hstack([image for image in Imagelist])
    
    if plot_flag == 1:
        fig, ax = plt.subplots(1)
        ax.imshow(imageReconstruct, cmap = 'gray')
    
    # build out the components of a confusion matrix
    n00, n01, n10, n11 = 0, 0, 0, 0 
               
    # Add on classification box rectangles
    for i, _ in enumerate(Imagelist):
        
        # Get stats on the current image
        if WP_io[i] == 0:
            if classification_result[i] == 0:
                n00 += 1
            if classification_result[i] == 1:
                n01 += 1 
        elif WP_io[i] == 1:
            if classification_result[i] == 0:
                n10 += 1
            if classification_result[i] == 1:
                n11 += 1
        
        # Add in the classification guess
        if classification_result[i] == 1 & plot_flag==1:
            rect = Rectangle((i*slice_width, 5), slice_width, height-10,
                                     linewidth=0.5, edgecolor='red', facecolor='none')
            ax.add_patch(rect)  
            
        if plot_flag==1:
            # Adds a rectangle for the confidence of classification at every square
            prob = confidence[i,0]
            rect = Rectangle((i*slice_width, 5), slice_width, height-10,
            linewidth=1.0*prob*prob, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(i*slice_width, height+70,round(prob,2), fontsize = 7)
            
            
    # Compute the inter-image accuracy
    acc = (n00 + n11) / (n00 + n11 + n10 + n01)
    print(f'Image {i_iter}: accuracy = {acc}')
    
    # Save off data for whole-set analysis
    TP_history.append(n11)
    TN_history.append(n00)
    FP_history.append(n01)
    FN_history.append(n10)
    acc_history.append(acc)
    confidence_history.append(confidence)
    WP_io_history.append(WP_io)
    
    if plot_flag == 1:
        # Check if there's even a bounding box in the image
        if sm_bounds[0] == 'X':
            ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN class - 32 nodes')
            plt.show()
            #print(f'i_iter: {i_iter} - No WP')
            continue
        else:
            # Add the ground truth over the entire box
            ax.add_patch(Rectangle((sm_bounds[0],sm_bounds[1]), sm_bounds[2], sm_bounds[3], edgecolor='blue', facecolor='none'))
        
            ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN class - 32 nodes')
            plt.show()
            #print(f'i_iter: {i_iter} - WP')

#%% Make history plot

# Take a rolling average
def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

Nframe_per_img = len(Imagelist)
n = 20 # Moving avg window
    
fig, (pl1, pl2, pl3, pl4, pl5) = plt.subplots(5,1, figsize = (16,16))
pl1.plot(range(len(acc_history)), acc_history)
pl1.plot(range(n-1, len(acc_history)), moving_average(acc_history, n), color='k', linewidth = 2)
pl1.set_title('Accuracy')

pl2.plot(range(len(TP_history)), [img_stat/Nframe_per_img for img_stat in TP_history])
pl2.plot(range(n-1, len(acc_history)), moving_average(TP_history, n)/Nframe_per_img, color='k', linewidth = 2)
pl2.set_title('True positive rate')

pl3.plot(range(len(TN_history)), [img_stat/Nframe_per_img for img_stat in TN_history])
pl3.plot(range(n-1, len(TN_history)), moving_average(TN_history, n)/Nframe_per_img, color='k', linewidth = 2)
pl3.set_title('True negative rate')

pl4.plot(range(len(FP_history)), [img_stat/Nframe_per_img for img_stat in FP_history])
pl4.plot(range(n-1, len(FP_history)), moving_average(FP_history, n)/Nframe_per_img, color='k', linewidth = 2)
pl4.set_title('False positive rate')

pl5.plot(range(len(FN_history)), [img_stat/Nframe_per_img for img_stat in FN_history])
pl5.plot(range(n-1, len(FN_history)), moving_average(FN_history, n)/Nframe_per_img, color='k', linewidth = 2)
pl5.set_title('False negative rate')


#%% Compute MICE
Nframe_per_img = len(Imagelist)
MICE = []
for i in range(len(acc_history)):
    n0 = TP_history[i] + FP_history[i]
    n1 = TN_history[i] + FN_history[i]
    
    A0 = (n0/Nframe_per_img)**2 + (n1/Nframe_per_img)**2
    if np.isclose(1-A0, 0):
        MICE.append(0.0)
    else:
        MICE.append((acc_history[i] - A0)/(1-A0))
    
    
fig, ax = plt.subplots(1,1, figsize = (16,6))
ax.plot(range(len(MICE)), MICE)
ax.plot(range(n-1, len(MICE)), moving_average(MICE, n), color='k', linewidth = 2)
ax.set_ylim(-1,1)
ax.set_title('MICE Performance')


#%% Print out the entire data set statistics
print("Data set statistics")
print("----------------------------------------")
print(f"Whole-set Average: {np.mean(acc_history)}")
print(f"Whole-set True Positive rate: {np.mean(TP_history)/Nframe_per_img}")
print(f"Whole-set True Negative rate: {np.mean(TN_history)/Nframe_per_img}")
print(f"Whole-set False Positive rate: {np.mean(FP_history)/Nframe_per_img}")
print(f"Whole-set False Negative rate: {np.mean(FN_history)/Nframe_per_img}")
print(f"Whole-set MICE Score: {np.mean(MICE)}")


#%% Form an ROC curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
thresholds = np.linspace(0, 1, num=50)
TPRs, FPRs, Pres = [], [], []
# Loop thru the thresholds
for threshold in thresholds:
    TP, FP, TN, FN = 0, 0, 0, 0
    
    # Loop thru each image in the test set
    for i in range(len(acc_history)):
        
        # Pull off the sliced list
        WP_io_img = WP_io_history[i]
        confid_img = confidence_history[i]
        slice_classification = []
        
        # Form the classification list under the new thrshold
        n00, n01, n10, n11 = 0, 0, 0, 0 
        for j in range(len(WP_io_img)):
            if confid_img[j] > threshold:
                slice_classification.append(1)
            else:
                slice_classification.append(0)
                
            # Now compute the TPR/FPR of the frame
            if WP_io_img[j] == 0:
                if slice_classification[j] == 0:
                    n00 += 1
                if slice_classification[j] == 1:
                    n01 += 1 
            elif WP_io_img[j] == 1:
                if slice_classification[j] == 0:
                    n10 += 1
                if slice_classification[j] == 1:
                    n11 += 1
        
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



