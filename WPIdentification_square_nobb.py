# Version 2 of a CNN used to identify second-mode waves using ResNet50 for feature extraction

import numpy as np
import os
import time
import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import tensorflow as tf

from keras.applications import resnet50
from keras.models import Model

# custom libraries
from processdata import write_data, image_splitting, whole_image
from stats import model_stats, whole_set_stats
from classify import classify_the_images, classify_the_frame
from modelfunctions import get_bottleneck_features, resnet_feature_extractor, feature_extractor_training

#%% Be able to run Second-Mode Wave detection, Turbulence detection, or both 
#(both defaults to using Second-Mode Wave detection dataset for labeling and whole-set statistics)

second_mode = True
sm_file_name = "C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\wavepacket_labels_combined.txt"
sm_N_img = 10
if second_mode:
    print('Finding second-mode waves')

#turbulence currently does not do post-processing
turb = False
turb_file_name = "C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\Test1\\run33\\turbulence_training_data.txt"
turb_N_img = 200
if turb:
    print('Finding turbulence')
    
whole_set_file_name = "C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\110000_111000_decimateby1\\Test1\\run33\\video_data.txt"
#"C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\110000_111000_decimateby1\\Test1\\run33\\video_data.txt"

slice_width = 64
ne = 20

plot_flag = 0      # View the images? MUCH SLOWER (view - 1, no images - 0)
N_frames = -1      # Number of frames to go through for whole-set
                    # If you want the whole-set -> N_frames = -1

# Calculate approx how many pixels a wave will propagate in a single frame
mm_pix = 0.0756        # From paper, mm to pixel conversion
FR = 258e3                 # Camera frame rate in Hz
dt = 1/FR                     # Time step between frames
prop_speed = 825       # A priori estimate of propagation speed, m/s
pro_speed_pix_frame = prop_speed * dt * 1/(mm_pix*1e-3)  # Computed number of pixels traveled between frames


if not second_mode and not turb:
    raise ValueError('One or both of "second_mode" and "turb" must be true')
    
#%% Split the test and train images and feature extract
start_time = time.time()

resnet_model = resnet_feature_extractor()

if second_mode:
    trainimgs, testimgs, trainlbs, testlbls, lines_len = write_data(sm_file_name, sm_N_img, slice_width)
    
    trainimgs_res = get_bottleneck_features(resnet_model, trainimgs, verbose = 1)
    testimgs_res = get_bottleneck_features(resnet_model, testimgs, verbose = 1)
if turb:
    trainimgs_turb, testimgs_turb, trainlbs_turb, testlbls_turb, lines_len_turb = write_data(turb_file_name, turb_N_img, slice_width)
    
    trainimgs_res_turb = get_bottleneck_features(resnet_model, trainimgs_turb, verbose = 1)
    testimgs_res_turb = get_bottleneck_features(resnet_model, testimgs_turb, verbose = 1)

#%% Call fcn to train the model!
if second_mode:
    history, model, ne = feature_extractor_training(trainimgs_res, trainlbs, testimgs_res, ne)
    print("Second-mode Wave Model Training Complete!")

if turb:
    history_turb, model_turb, ne_turb = feature_extractor_training(trainimgs_res, trainlbs, testimgs_res, ne)
    print("Turbulence Model Training Complete!")

end_time = time.time()
print(f'Time to read in data, process, and train model: {end_time-start_time} seconds')

#%% Perform the visualization
'''
Visualization: inspect how the training went
'''
    
#%% Show results
if second_mode:
    model_stats(ne,history,model,testimgs_res,testlbls,'Second-mode')

if turb:
    model_stats(ne_turb,history_turb,model_turb,testimgs_res_turb,testlbls_turb,'Turbulence')

#%% Save off the model, if desired
#model.save('C:\\Users\\tyler\\Desktop\\NSSSIP25\\TrainedModels\\run33_strangelyhighvalacc_95.keras')
#model_turb.save('C:\\Users\\tyler\\Desktop\\NSSSIP25\\TrainedModels\\turb_model_June24_bad.keras')
#print('Model(s) saved')

#%% Classification code - stats only for file_name labels!!! (not other classes)

#%% Read image and existing classification model into the workspace
print('Reading training data file')

# Write File Name
if os.path.exists(whole_set_file_name):
    with open(whole_set_file_name, 'r') as file:
        lines = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len = len(lines)
print(f"{lines_len} lines read")

# Transfer learning model for stacking on ResNet50
model1 = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
output = model1.output
output = tf.keras.layers.Flatten()(output)
resnet_model = Model(model1.input,output)

#no need to load in model - already exists

#%% Iterate through the list!
# Initialize some arrays for later data analysis
N_img = lines_len
acc_history = []
TP_history = []
TN_history = []
FP_history = []
FN_history = []
WP_io_history = []
confidence_history = []
filtered_result_history = []
classification_history = []


plot_flag = plot_flag       # View the images? MUCH SLOWER
window_size = 3             # Moving window to filter the frames
indiv_thres = 0.85          # Individual exception threshold
confid_thres = 1.5          # SUMMED confidence over the entire window. 
                            # e.g. for 0.5 over 3 windows, make this value 1.5
use_post_process = 1        # 1 to use windowing post process, 0 if not

if N_frames == -1:
    N_frames = N_img

print(f'Classifying {N_frames} images (this may take a while)...')

### Iterate over all frames in the video
for i_iter in range(N_frames):
    
    ### Perform the classification
    
    if not second_mode:
        model = 0 #useless input
    if not turb:
        model_turb = 0 #useless input to satisfy classify_the_images inputs if not finding turbulence
    
    # Split the image and classify the slices
    Imagelist, Imagelist_res, WP_io, slice_width, height, sm_bounds = image_splitting(i_iter, lines, slice_width)
        
    if second_mode:
        simple_class_result, confidence = classify_the_images(model, resnet_model, Imagelist_res)
    else:
        simple_class_result, confidence = classify_the_images(model_turb, resnet_model, Imagelist_res)
        
    # Analyze and filter the image results
    classification_result, filtered_result, n00, n01, n10, n11 = classify_the_frame(Imagelist,WP_io, confidence, window_size, indiv_thres, confid_thres, model_turb,Imagelist_res,i_iter, lines, slice_width, second_mode, turb, use_post_process)
    
    
    ### Restitch and display the classification results
    # Restack and plot the image
    imageReconstruct = np.hstack([image for image in Imagelist])
    if plot_flag == 1:
        fig, ax = plt.subplots(1)
        ax.imshow(imageReconstruct, cmap = 'gray')
        
        if second_mode:
            ax.text(-45, height+86,'WP: ', fontsize = 6)
        if turb:
            ax.text(-57, height+62,'Turb: ', fontsize = 6)

        
        # Add on classification box rectangles
        for i, _ in enumerate(Imagelist):    
            # Add in the classification guess
            if classification_result[i] == 1:
                rect = Rectangle((i*slice_width, 5), slice_width, height-10,
                                         linewidth=0.5, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            elif classification_result[i] >= 2:
                ax.text(i*slice_width+slice_width/5, height+62,f'{(classification_result[i]-2):.2f}', fontsize = 6)
                if classification_result[i]-2 >= 0.5:
                    rect = Rectangle((i*slice_width, 5), slice_width, height-10,
                                             linewidth=0.5, edgecolor='orange', facecolor='none')
                    ax.add_patch(rect)
            if second_mode:
                prob = round(confidence[i,0],2)
                ax.text(i*slice_width+slice_width/5, height+86,f'{prob:.2f}', fontsize = 6)
            
                
        '''
            # Adds a rectangle for the confidence of classification at every square
            prob = confidence[i,0]
            rect = Rectangle((i*slice_width, 5), slice_width, height-10,
            linewidth=1.0*prob*prob, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(i*slice_width, height+60,round(prob,2), fontsize = 7)
            '''
            
            
    ### Save off the classification results for later analysis
    acc = (n00 + n11) / (n00 + n11 + n10 + n01)
    TP_history.append(n11)
    TN_history.append(n00)
    FP_history.append(n01)
    FN_history.append(n10)
    acc_history.append(acc)
    confidence_history.append(confidence)
    WP_io_history.append(WP_io)
    filtered_result_history.append(filtered_result)
    classification_history.append(classification_result)
    
    
    ### Finally, plot the ground truth
    if plot_flag == 1:
        # Check if there's even a bounding box in the image
        if sm_bounds[0] == 'X':
            if second_mode and turb:
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN WP. Orange: NN Turbulence')
            elif second_mode:
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN WP')
            else:
                ax.set_title('Image '+str(i_iter)+'. Blue: Labeled Turbulence')
            plt.show()
            continue
        else:
            # Add the ground truth over the entire box
            ax.add_patch(Rectangle((sm_bounds[0],sm_bounds[1]), sm_bounds[2], sm_bounds[3], edgecolor='blue', facecolor='none'))
            if second_mode and turb:
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN WP. Orange: NN Turbulence')
            elif second_mode:
                ax.set_title('Image '+str(i_iter)+'. Blue: true WP. Red: NN WP')
            else:
                ax.set_title('Image '+str(i_iter)+'. Blue: Labeled Turbulence')
            plt.show()

print('Done classifying the video!')

#%% Run whole-set stats
sum1=0
for n,_ in enumerate(WP_io_history):
   sum1+=sum(WP_io_history[n])
   
if sum1 !=0: #checks to see if bounding boxes exist
    if second_mode:
        print('Whole-set stats based on Second-mode Waves')
        whole_set_stats(Imagelist,acc_history,TP_history,TN_history,FP_history,FN_history,WP_io_history,confidence_history)
    else:
        print('Whole-set stats based on Turbulence')
        whole_set_stats(Imagelist,acc_history,TP_history,TN_history,FP_history,FN_history,WP_io_history,confidence_history)
else:
    print('Sum of WP_io_history == 0')
    print('Implies that whole-set data has no bounding boxes')

#%% Simple breakdown code
# if a slice is turbulence check if the slice before it in the previous frame was a WP
# assumes waves propogate at one slice per frame interval

where = [] #store where the breakdown occurs
    
if second_mode and turb: 
    ### Iterate over all frames in the video
    for i_iter in range(N_frames):
        if i_iter == 0: #skips the first frame (can't go back in time to check breakdown)
            continue
        for i, _ in enumerate(Imagelist):
            if i == 0: #skips the first slice (can't go back in space to check breakdown)
                continue
            if (classification_history[i_iter][i])-2 > 0.5: #if turbulent...
                if classification_history[(i_iter-1)][(i-1)]: #look back one in time and space to see it a WP preceeded it
                    where.append(i)
    plt.hist(where)
    plt.show()
    
#%% More complex breakdown statistics

#can maybe add code later to do unit propagation speed conversion from m/s or something comparable

thres = 0.5
turb_detect_count = 0
turb_count = 0
from_count = 0
dis_trav = []
wait = 2
preserve_classification_history = copy.deepcopy(classification_history) #saves a deep copy before overwriting later
#%%
#use to reset: classification_history = copy.deepcopy(preserve_classification_history)

if second_mode and turb: 
    #find total number of slices detected as turbulence
    for i_iter in range(N_frames-1,0,-1):
        for i, _ in enumerate(Imagelist):
            if (classification_history[i_iter][i])-2 > thres:
                turb_detect_count = turb_detect_count + 1 
                
    #find breakdown stats
    ### Iterate over all frames in the video
    for i_iter in range(N_frames-1,0,-1): #goes through frames backwards; stops before first image
        overwrite = []
        for i, _ in enumerate(reversed(Imagelist)): #goes through images starting on the right
            if i <= pro_speed_pix_frame//slice_width-1: #skips the first slice (or multiple if pro. speed is high enough) -- (can't go back in space to check breakdown)
                continue
            if (classification_history[i_iter][i])-2 > thres: #if turbulent...
                from_second_mode = False
                turb_count = turb_count + 1
                i_loop = 1
                patience = wait #gives additional chance(s) if not detected the first time
                while patience >= 1:
                    if classification_history[(i_iter-i_loop)][int(i-pro_speed_pix_frame*i_loop//slice_width)]-2 > thres: #check for preceeding turbulence
                        #classification_history[(i_iter-i_loop)][int(i-pro_speed_pix_frame*i_loop//slice_width)] = 0 #overwrite to 0 to avoid double counting turbulence
                        overwrite.append([i_iter-i_loop, int(i-pro_speed_pix_frame*i_loop//slice_width)])
                        if i-pro_speed_pix_frame*i_loop//slice_width > 0 and i_iter-i_loop>=0: #check to avoid exceeding image bounds and first frame
                            i_loop = i_loop + 1
                        else:
                            patience = 0 
                            break
                        if patience < wait:
                            patience = wait #reset additional chance(s) if WP is detected
                    elif classification_history[(i_iter-i_loop)][int(i-pro_speed_pix_frame*i_loop//slice_width)] == 1: #is it a WP already?
                        break
                    else:
                        patience = patience - 1
                        i_loop = i_loop + 1
                if patience<wait: #don't count additional chance(s) that found nothing
                    i_loop = i_loop+patience-wait
                mark = i_loop
                patience = wait
                while patience >= 1:
                    if classification_history[(i_iter-i_loop)][int(i-pro_speed_pix_frame*i_loop//slice_width)] == 1:
                        if i-pro_speed_pix_frame*i_loop//slice_width > 0 and i_iter-i_loop>=0: #check to avoid exceeding image bounds and first frame
                            i_loop = i_loop + 1
                        else:
                            break
                        if patience < wait:
                            patience = wait #reset additional chance(s) if WP is detected
                        from_second_mode = True
                    else:
                        patience = patience - 1
                        i_loop = i_loop + 1
                if patience<wait: #don't count additional chance(s) that found nothing
                    i_loop = i_loop+patience-wait
                if from_second_mode:
                    from_count = from_count + 1
                if i_loop != mark: #checks if turbulence actually came from observed wave packet
                    dis_trav.append((i_loop-mark)*pro_speed_pix_frame)
        for n_i, n_iter in overwrite: #over write at the end of the loop to protect overlaps
            classification_history[(n_i)][n_iter] = 0
    
    print(f'Total number of turbulence slices detected: {turb_detect_count}')           
    print(f'Turbulence count (duplicates overwritten): {turb_count}')
    if turb_detect_count == turb_count:
        print('If above are equal, duplicates have likely already been removed or dataset is too small or laminar')
    print(f'Percentage of turbulence observed to develop from WPs: {round(from_count/turb_count*100,2)}%')
    print(f'Mean distance traveled: {round(np.mean(dis_trav),2)} pixels')
    print(f'Median distance traveled: {round(np.median(dis_trav),2)} pixels')
    plt.hist(dis_trav)
    plt.title('Distance WP traveled before breaking down into turbulence')
    plt.ylabel('Number of WPs')
    plt.xlabel('Distance traveled (pixels)')
    plt.show()
    
#%% Breakdown visualization

max_i_loop = 0
cls_his = copy.deepcopy(preserve_classification_history)
first_turb_frame = True
store_loc = [] #store relevent slices to add boxes to later 

if second_mode and turb:
    for i_iter in range(N_frames-1,0,-1): #goes through frames backwards; stops before first image
        if first_turb_frame == False:
            i_iter_record = i_iter + 1
            break
        else:
            store_loc = [] #clears if turbulence did not come from wavepacket
        overwrite = []
        for i, _ in enumerate(reversed(Imagelist)): #goes through images starting on the right
            if i <= pro_speed_pix_frame//slice_width-1: #skips the first slice (or multiple if pro. speed is high enough) -- (can't go back in space to check breakdown)
                continue
            if (cls_his[i_iter][i])-2 > thres: #if first frame with turbulence
                store_loc.append([i_iter,i,2]) # 2 is for turbulence
                i_loop = 1
                patience = wait #gives additional chance(s) if not detected the first time
                while patience >= 1:
                    if cls_his[(i_iter-i_loop)][int(i-pro_speed_pix_frame*i_loop//slice_width)]-2 > thres: #check for preceeding turbulence
                        overwrite.append([i_iter-i_loop, int(i-pro_speed_pix_frame*i_loop//slice_width)])
                        if i-pro_speed_pix_frame*i_loop//slice_width > 0 and i_iter-i_loop>=0: #check to avoid exceeding image bounds and first frame
                            store_loc.append([i_iter-i_loop,int(i-pro_speed_pix_frame*i_loop//slice_width),2]) # 2 is for turbulence
                            i_loop = i_loop + 1
                        else:
                            patience = 0 
                            break
                        if patience < wait:
                            patience = wait #reset additional chance(s) if WP is detected
                    elif classification_history[(i_iter-i_loop)][int(i-pro_speed_pix_frame*i_loop//slice_width)] == 1: #is it a WP already?
                        break
                    else:
                        patience = patience - 1
                        i_loop = i_loop + 1
                if patience<wait: #don't count additional chance(s) that found nothing
                    i_loop = i_loop+patience-wait
                patience = wait #gives additional chance(s) if not detected the first time
                while patience >= 1:
                    if cls_his[(i_iter-i_loop)][int(i-pro_speed_pix_frame*i_loop//slice_width)] == 1:
                        first_turb_frame = False #only want propagation that starts from WP and ends in one frame
                        if i-pro_speed_pix_frame*i_loop//slice_width > 0 and i_iter-i_loop>=0: #check to avoid exceeding image bounds and first frame
                            i_loop = i_loop + 1
                            store_loc.append([i_iter-i_loop,int(i-pro_speed_pix_frame*i_loop//slice_width),1]) # 1 is for WP
                        else:
                            break
                        if patience < wait:
                            patience = wait #reset additional chance(s) if WP is detected
                    else:
                        patience = patience - 1
                        i_loop = i_loop + 1
                if patience<wait: #don't count additional chance(s) that found nothing
                    i_loop = i_loop+patience-wait
                if i_loop > max_i_loop:
                    max_i_loop = i_loop
        for n_i, n_iter in overwrite: #over write at the end of the loop to protect overlaps
            classification_history[(n_i)][n_iter] = 0
    
    fig, axs = plt.subplots(max_i_loop+1)
    fig.suptitle(f'Turbulence and Wavepacket Breakdown\n Red: NN WP. Orange: NN Turbulence\nImages {i_iter_record-max_i_loop} to {i_iter_record}')
    for n in range(max_i_loop+1):
        axs[n].imshow(whole_image(n,lines),cmap = 'gray')
    for m,_ in enumerate(store_loc):
        if store_loc[m][2] == 1:
            color = 'red'
        else:
            color = 'orange'
        rect = Rectangle((store_loc[m][1]*slice_width, 5), slice_width, height-10,
                                 linewidth=0.5, edgecolor=color, facecolor='none')
        axs[store_loc[m][0]-i_iter_record+max_i_loop].add_patch(rect)
    
    #axs[max_i_loop-1].text(store_loc[1][1]*slice_width, 30,round(cls_his[store_loc[max_i_loop-1][0]][store_loc[max_i_loop-1][1]],2), fontsize = 8)
            
    #ax.text(i*slice_width, height+60,round(prob,2), fontsize = 7)

    
    plt.show()

