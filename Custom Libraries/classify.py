import numpy as np


def calc_windowed_confid(j, confidence, window_size):
    '''
    Calculates the local confidence (i.e. a single slice of a frame) 
    via a summed windowing method
    '''
    if (j - window_size//2) < 0: # at the front end of the image
        local_confid = np.sum(confidence[0:j+window_size//2+1:1])
    elif (j + window_size//2) > len(confidence): # at the end of the image list
        local_confid = np.sum(confidence[j-window_size//2-1:len(confidence):1])
    else:
        local_confid = np.sum(confidence[j-window_size//2:j+window_size//2+1:1])
        
    return local_confid

def classify_the_images(model, Imagelist_res, verbose=0): #not verbose by default
        # Pass each through the trained NN
        test_res = model.predict(Imagelist_res, verbose = 0)
        classification_result = np.round(test_res)
        return classification_result, test_res, Imagelist_res

def classify_the_frame(Imagelist,WP_io, confidence, window_size, indiv_thres, confid_thres, model_turb,Imagelist_res,i_iter, lines, slice_width, second_mode, turb, use_post_process):
    n00, n01, n10, n11 = 0, 0, 0, 0 
    filtered_result = []
    classification_result = np.zeros(len(Imagelist))
        
    if second_mode:
        for i, _ in enumerate(Imagelist):
            # If using the windowed post processing, call the windowing fcn
            # to get the locally informed confidence. Then compare to thresholds
            if use_post_process == 1:
                local_confid = calc_windowed_confid(i, confidence, window_size)
                
                # Are window and indiv conditions met?
                if (local_confid > confid_thres) or (confidence[i] > indiv_thres):
                    filtered_result.append(1)
                elif turb: #if we are also checking for turbulence
                    #Imagelist_resized = img_preprocess(Imagelist[i])
                    # Run through feature extractor
                    #Imagelist_res = get_bottleneck_features(resnet_model, Imagelist_resized)
                    test_res = model_turb.predict(Imagelist_res[i:i+1],verbose=0) #checks for turbulence
                    if test_res < 0:
                        filtered_result.append(0)
                    else:
                        filtered_result.append(2+test_res)
                else:
                    filtered_result.append(0)
                
                classification_result[i] = filtered_result[i]
            
            # If not, then just round
            else:
                check = np.round(confidence[i])
                if check == 0 and turb:
                    #Imagelist_resized = img_preprocess(Imagelist[i])
                    # Run through feature extractor
                    #Imagelist_res = get_bottleneck_features(resnet_model, Imagelist_resized)
                    test_res = model_turb.predict(Imagelist_res[i:i+1],verbose=0) #checks for turbulence
                    if test_res < 0.5:
                        classification_result[i] = 0
                    else:
                        classification_result[i] = 2+test_res
                elif check == 0:
                    classification_result[i] = 0
                else:
                    classification_result[i] = check
                
            # Get stats on the current image
            if WP_io[i] == 0:
                if classification_result[i] != 1:
                    n00 += 1
                if classification_result[i] == 1:
                    n01 += 1 
            elif WP_io[i] == 1:
                if classification_result[i] != 1:
                    n10 += 1
                if classification_result[i] == 1:
                    n11 += 1
    else: #for turbulence only
        for i, _ in enumerate(Imagelist):
            test_res = confidence[i]
            if test_res < 0.5:
                classification_result[i] = 0
            else:
                classification_result[i] = 2+test_res
            
            # Get stats on the current image
            if WP_io[i] == 0:
                if classification_result[i] == 0:
                    n00 += 1
                if classification_result[i] != 0:
                    n01 += 1 
            elif WP_io[i] == 1:
                if classification_result[i] == 0:
                    n10 += 1
                if classification_result[i] != 0:
                    n11 += 1                
            
                
    return classification_result, filtered_result, n00, n01, n10, n11