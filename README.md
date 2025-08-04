# 2025_Hypersonic_BL_ID
Hypersonic boundary layer feature identification using ML-enabled components

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
