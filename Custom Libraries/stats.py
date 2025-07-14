import numpy as np
import matplotlib.pyplot as plt



def model_stats(ne,history,model,testimgs_res,testlbls,name):
    #model.save('ClassifierV1m.h5')
    epoch_list = list(range(1,ne + 1))
    # Making some plots to show our results
    f, (pl1, pl2) = plt.subplots(1, 2, figsize = (15,4), gridspec_kw = {'wspace': 0.3})
    f.suptitle('Neural Network Performance: ' + name, fontsize = 14)
    # Accuracy Plot
    pl1.plot(epoch_list, history.history['accuracy'], label = 'train accuracy')
    pl1.plot(epoch_list, history.history['val_accuracy'], label = 'validation accuracy')
    pl1.set_xticks(np.arange(0, ne + 1, 5))
    pl1.set_xlabel('Epoch')
    pl1.set_ylabel('Accuracy')
    pl1.set_title('Accuracy')
    pl1.legend(loc = "best")
    # Loss plot for classification
    pl2.plot(epoch_list, history.history['loss'], label = 'train loss')
    pl2.plot(epoch_list, history.history['val_loss'], label = 'validation loss')
    pl2.set_xticks(np.arange(0, ne + 1, 5)) 
    pl2.set_xlabel('Epoch')
    pl2.set_ylabel('Loss')
    pl2.set_title('Classification Loss')
    pl2.legend(loc = "best")
    plt.show()
    
    # Implement some statistics
    
    # Check how well we did on the test data!
    test_res= model.predict(testimgs_res)
    test_res_binary = np.round(test_res)
    
    # build out the components of a confusion matrix
    n00, n01, n10, n11 = 0, 0, 0, 0 
    
    for i, label_true in enumerate(testlbls):
        label_pred = test_res_binary[i]
        
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
           
    n0 = n00 + n01
    n1 = n10 + n11
    
    # Compute accuracy, sensitivity, specificity, 
    # positive prec, and neg prec
    # As defined in:
        # Introducing Image Classification Efficacies, Shao et al 2021
        # or https://arxiv.org/html/2406.05068v1
        # or https://neptune.ai/blog/evaluation-metrics-binary-classification
        
    TP = n11
    TN = n00
    FP = n01
    FN = n10
        
    acc = (n00 + n11) / len(testlbls) # complete accuracy
    Se = n11 / n1 # true positive success rate, recall
    Sp = n00 / n0 # true negative success rate
    Pp = n11 / (n11 + n01) # correct positive cases over all pred positive
    Np = n00 / (n00 + n10) # correct negative cases over all pred negative
    Recall = TP/(TP+FN) # Probability of detection
    FRP = FP/(FP+TN) # False positive, probability of a false alarm
    
    # Rate comapared to guessing
    # MICE -> 1: perfect classification. -> 0: just guessing
    A0 = (n0/len(testlbls))**2 + (n1/len(testlbls))**2
    MICE = (acc - A0)/(1-A0)   
    
    # Print out the summary statistics
    ntot = len(testlbls)
    print("---------" + name + " Test Results---------")
    print("            Predicted Class         ")
    print("True Class     0        1    Totals ")
    print(f"     0        {n00}       {n01}    {n0}")
    print(f"     1        {n10}        {n11}    {n1}")
    print("")
    print("            Predicted Class         ")
    print("True Class     0        1    Totals ")
    print(f"     0        {n00/ntot}      {n01/ntot}    {n0}")
    print(f"     1        {n10/ntot}      {n11/ntot}    {n1}")
    print("")
    print(f"Model Accuracy: {acc}, Sensitivity: {Se}, Specificity: {Sp}")
    print(f"Precision: {Pp},  Recall: {Recall}, False Pos Rate: {FRP}")
    print(f"MICE (0->guessing, 1->perfect classification): {MICE}")
    print("")
    print(f"True Pos: {n11}, True Neg: {n00}, False Pos: {n01}, False Neg: {n10}")

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def whole_set_stats(Imagelist,acc_history,TP_history,TN_history,FP_history,FN_history,WP_io_history,confidence_history):
    
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
    
    
    # Compute MICE
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
    
    
    # Print out the entire data set statistics
    print("Data set statistics")
    print("----------------------------------------")
    print(f"Whole-set Average: {np.mean(acc_history)}")
    print(f"Whole-set True Positive rate: {np.mean(TP_history)/Nframe_per_img}")
    print(f"Whole-set True Negative rate: {np.mean(TN_history)/Nframe_per_img}")
    print(f"Whole-set False Positive rate: {np.mean(FP_history)/Nframe_per_img}")
    print(f"Whole-set False Negative rate: {np.mean(FN_history)/Nframe_per_img}")
    print(f"Whole-set MICE Score: {np.mean(MICE)}")
    
    
    # Form an ROC curve
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
                    if slice_classification[j] != 1:
                        n00 += 1
                    if slice_classification[j] == 1:
                        n01 += 1 
                elif WP_io_img[j] == 1:
                    if slice_classification[j] != 1:
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