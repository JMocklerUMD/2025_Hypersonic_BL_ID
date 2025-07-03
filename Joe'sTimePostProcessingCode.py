

#%% Filter the ID's in time

window_size = 3
confid_thres = 1.5
indiv_thres = 0.9
use_post_process = 1
WP_locs_list = []

for i in range(1995):
    # Start by classifying i'th frame
    Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i, lines)
    confidence = Confidence_history[i]
    classification_result, filtered_result, n00, n01, n10, n11 = classify_the_frame(Imagelist, confidence, WP_io, window_size, indiv_thres, confid_thres, use_post_process)
   
    # Perform a "lookahead" of the next 3 frames
    # Sum the classifications along the time dimension
    # (e.g form a 19x1 array that's the sum 4 time steps of classification results)
    WP_locs = np.zeros(len(Imagelist))
    WP_locs = WP_locs + filtered_result
    for ii in range(1,4):
        Imagelist, WP_io, slice_width, height, sm_bounds = image_splitting(i + ii, lines)
        confidence = Confidence_history[i+ii]
        classification_result, filtered_result, n00, n01, n10, n11 = classify_the_frame(Imagelist, confidence, WP_io, window_size, indiv_thres, confid_thres, use_post_process)
       
        # Sum in time
        WP_locs = WP_locs + filtered_result
       
    # Now go thru the list and ID where, among the 4 frames, there's consistency
    # NOTE: change the confidence level below (WP_candidate > 3) to a higher or
    # for a different threshold
    start_slice = 0
    stop_slice = 0
    consec = 0
    for j, WP_candidate in enumerate(WP_locs):
        # Determine when we first detect a WP moving left-right
        if WP_candidate > 3 and consec == 0:
            start_slice = j
            consec = 1
       
        # Now see when the WP stops
        if WP_candidate < 2 and consec == 1:
            consec = 0
            stop_slice = j
           
            # Perform a correction to account for left-right convection
            if WP_locs[start_slice-1] > 0:
                start_slice = start_slice - 1
               
            # Handle when the WP is at the start of the frame
            if start_slice < 0:
                start_slice = 0
                WP_locs_list.append([start_slice, stop_slice])
            break # break the enumerate(WP_locs) for loop
   
        # Handle advection off the screen
        if consec == 1 and j == len(WP_locs) - 1:
            stop_slice = j
   
    # For a list of start-stop slices for the entire analyzed video set
    WP_locs_list.append([start_slice, stop_slice])        
   
    # Print and inspect if you want
    #print(f"Frame {i}: WP_loc = [{start_slice}, {stop_slice}]")   