# Hypersonic Boundary Layer Structural Identification
This code-base uses Keras-implementation transfer learning to automatically identify and localize the existence of second-mode instabilities and turbulence in schlieren video data. We recommend this code for experimental hypersonic boundary layer researchers who are interested in analyzing the statistical properties of such instabilities. Additionally, we have only demonstrated results for second-mode (Mack mode) and fully developed turbulence, but note that first-mode or more transitionary features (such as nonlinear second-mode growth, etc.) could be detected using a similar approach.

We encourage users to read [our paper](https://github.com/JMocklerUMD/2025_Hypersonic_BL_ID/blob/main/USJH_Hypersonic_BL_ID_Preprint.pdf) first.

This project was completed in conjunction with the Army Research Lab, Aberdeen Proving Ground. 

Authors: Joseph Mockler (jmockle1@umd.edu), Catherine Stock (cstock1@terpmail.umd.edu), Chase Latyak (rclatyak@terpmail.umd.edu), Tyler Ostrowski (tylerost@terpmail.umd.edu)

![/ReadMe Figures/WP_prop.png](https://github.com/JMocklerUMD/2025_Hypersonic_BL_ID/blob/main/ReadMe%20Figures/WP_prop.png)

## Usage and Tutorial
Best-practice usage may be divided into two section: (1) model training and (2) model deployment. Software was developed and tested in spyder, a python IDE; if available, we recommend users run the software through this IDE for best experience.

### Model training
1. First, download "Example Data and Outputs.zip" from an associated author (above) and unzip the folder into the working directory
2. Gather a representative of experimental images to use for *model training*. A few considerations/best-practices:
   1. Training images should be representative of the video you want to classify. We recommend you randomly sample at least 200-300 frames from the section of video you seek to classify
   2. Data shouldn't be too sparse: ensure that the structures you want to classify are at least present in the random sample of images (minimum of 5% for each structure type). The model cannot classify structure's it hasn't seen!
   3. Label an extra ~100-200 frames to validate the data (discussed later).
   4. Only perform this analysis on straight body, schlieren image, hypersonic experiments.
3. Now run the **final_createTD.m** script to develop the *training data*
   1. Follow the printed instructs in the MATLAB console to label the video.
   2. Exiting will save your progress. If labelling over a long period, we recommend you exit and save often!
   3. Skip frames with MULTIPLE, seperated structures! You CANNOT label multiple boxes on a single frame; therefore, if you only label one structure, the other is implicitly labeled as NOT a structure. Similarly, if you label both and draw a box over regions of NO structure, you overlabel laminar flow!
   4. See **LangleyRun34_filtered_training_data.txt** and **LangleyRun34_turbulence_training_data.txt** for examples of such labeled text files. Additionally, if you want to just test the code, use these as labeled files to avoid gathering and labeling your own data.
   5. You'll have to repeat the labeling process for the types of structures you seek to classify (e.g. if you want to classify second-mode and turbulence, you'll need TWO labeled .txt files)
4. Inspect **ML_models.py** set model architecture and training. We recommend you keep this as-is, but you may change this to tune results or as your project is fit.
5. Run **Build_Classifier.py** to build your seperate classifiers
   1. First, pick a slice-size and ensure this is consistent throughout all analyses you perform. We recommend about 2-3x the wavelength (commonly 64 pixels), but let this vary by project. 
   2. You'll need one classifier for EACH type of structure you want to classify.
   3. Ensure training looks reasonable: accuracies plateau at near or above 90% and the loss is approximately minimized or at least monotonically decreasing. 
   4. Save the keras CNN models! If you're just running the script as an example, these are in **build_Classifier\secondmodemodel_LangleyRun34.keras** and **build_Classifier\turbulencemodel_LangleyRun34.keras**
6. If you choose to label additional 100-200 frames, you can test this in **Classify_a_video.py** (for just single structure type classification) or **Classify_a_video_SMandTurb.py.** Otherwise, your model is complete!

![/ReadMe Figures/WP_prop.png](https://github.com/JMocklerUMD/2025_Hypersonic_BL_ID/blob/main/ReadMe%20Figures/classified_ex.png)

### Model Deployment
1. Now classify a video of choosing. First, run **video_creation.m** to generate a similar .txt file (this time, without any labels) to read into our software. Some recommended best-practices:
   1. Use the *same slice size!* The classifier WILL NOT yield useful results unless the slice size used to train is the same here
   2. Use the *same bandpass filtering limits!*
   3. You can lightly tune the post-processing limits to get better estimates of locations. Review the paper for more precise definitions of what each does, but in general, larger values are more conservative estimates of structure localization and identification.
2. Save the classification results for more advanced analyses later
   1. Functionality is already embedded in the script, but this file should be a numpy array of lists corresponding to images (outer list) of classifications of each slice (inner lists)
   
### Advanced Analysis
While none of these are required for pure classification, they showcase some example analysis code we've already written and integrated with the classification results. Therefore, we won't present deep detail in each of these, but provide some general functions/guidance for each.
1. **Prop_speed_calculation.py** accepts the second-mode classification results, saved from the previous step, and computes the second-mode instability propagation speed through the BL. There are four example methods of performing the computation: 1D line-based correlation, 2D image-based correlation, quasi-2D based correlation, and 2D optical flow. NOTE: the images MUST be in sequential order!
  1. See if you can reproduce the histogram by training your own classifier, classifying an entire video, and running the prop speed calculator!
2. **Intermittency_plot_function.py** accepts the turbulence classification results and computes the turbulence intermittency downstream. 
3. **Visualize_breakdown_sequences.py (experimental)** uses combined classification results to identify sections of the video where the second-mode packet breaks down into fully-developed turbulence within the FOV.
We encourage researchers contribute additional analysis methods to build out our toolbox!

![(/ReadMe Figures/intermit_plot.png)](https://github.com/JMocklerUMD/2025_Hypersonic_BL_ID/blob/main/ReadMe%20Figures/intermit_plot.png)


## Files
- FOLDER: BL Norminalization Codes - see internal README

- FOLDER: Example Data and Outputs
  - Basic instructions and example data to run the following scripts:
    - Build_Classifier
    - Classify_a_video
    - Classify_a_video_SMandTurb
    - Prop_speed_calculation

- FOLDER: MATLAB Utility Codes - see internal README

- SCRIPT: Build_Classifier
  - input: .txt file for a labeled dataset, file pathing, parameters (specifically slice_width)
  - output: trained ML classifier, statistics, whole-set classified images (if desired)

- SCRIPT: Classify_a_video
  - input: .txt file with data (can be unlabeled), trained model, file pathing, parameters (specifically slice_width and, if applicable, bandpass and BL_height)
  - output: classified images (if desired), array with confidence history of classifications, statistics (if the dataset is labeled)

- SCRIPT: Classify_a_video_SMandTurb
  - input: .txt file with data (not bandpass filtered; can be unlabeled), second-mode wave packet trained model, turbulence trained model, file pathing, parameters (specifically slice_width BL_height)
  - output: classified images (if desired), array with confidence history of wave packet classifications, array of wave packet classifications, array of turbulence classifications, statistics (if the dataset is labeled)

- SCRIPT: Prop_speed_calculation
  - input: array with confidence history of wave packet classifications (from Classify_a_video or Classify_a_video_SMandTurb), file pathing, parameters (many - inspect code section by section)
  - output: propagation speed +/- one standard deviation from 1D and 2D correlation, charts to characterize propagation speed calculations, various images to visually inspect or display detection results, .mat file with array of measured propagation speeds, .mat file of sequence of images
  
- SCRIPT: Visualize_breakdown_sequences
  - input: array with confidence history of wave packet classifications (from Classify_a_video_SMandTurb), array of wave packet classifications (from Classify_a_video_SMandTurb), file pathing, parameters (specifically slice_width; see comments if not using a Langley dataset)
  - output: statistics on turbulence origin, sequences of images showing either wave packet breakdown or turbulence coming from upstream
  - NOTES: This code is EXPERIMENTAL and not yet deployable. It infrequently produces good results, but remains a work in progress.

- Custom FUNCTIONS:
  - ML_models
  - ML_utils
  - Results_utils
  - Propagation_speed_functions
  - intermittency_plot_function: not in any script; see internal comments for direction on use
  - Turbulence_and_breakdown_utils
