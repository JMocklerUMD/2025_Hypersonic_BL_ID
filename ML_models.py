# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 11:48:24 2025

@author: Joseph Mockler
"""

from ML_utils import *

from keras import optimizers, layers, regularizers
from keras.applications import resnet50 
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Model
from keras.models import Sequential
import keras

import tensorflow as tf
import numpy as np

from sklearn.utils import class_weight

import matplotlib.pyplot as plt


def feature_extractor_training(resnet_model, trainimgs_res, trainlbs, ne):
    """
    Building the Resnet50 model: a dense NN is trained on ResNet50 features to classify the existance and location of instabilites and turbulence.
    
    INPUTS
    ------------------
    trainimgs: (N, 224, 224, 3) numpy array of (224, 224, 3),
        image slices to train the model.
        
    trainlbs: (N,1) numpy array,
        binary classes
        
    testimgs: (M, 224, 224, 3) numpy array of (224, 224, 3),
        image slices to test the model.
        
    ne: int,
        number of epochs trained. Used for plotting later
    
    OUTPUTS: 
    ------------------
    history: keras NN model object,
        training history 
        
    model: keras NN model object,
        JUST the top-layer dense NN that accepts bottleneck features
        
    testimgs_res: (M, 100532) ResNet50 feature vector,
        One for each test image slice
        
    ne: int,
        number of epochs trained. Used for plotting later
    """
    
    print('Begin training the NN classification model')
    
    
    # Generate an input shape for our classification layers
    input_shape = resnet_model.output_shape[1]
    
    # Get unique classes and compute weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(trainlbs),
        y=trainlbs
        )
    
    # Convert to dictionary format required by Keras
    class_weights_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weights_dict)
    
    # Added the classification layers
    model = Sequential()
    model.add(InputLayer(input_shape = (input_shape,)))
    model.add(Dense(128,                                        # NN dimension            
                    activation = 'relu',                        # Activation function at each node
                    input_dim = input_shape,                    # Input controlled by feature vect from ResNet50
                    kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4),     # Regularization penality term
                    bias_regularizer=regularizers.L2(1e-4)))                    # Additional regularization penalty term
    
    model.add(Dropout(0.5))     # Add dropout to make the system more robust
    model.add(Dense(1, activation = 'sigmoid'))     # Add final classification layer
    
    # Compile the NN
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy'])
    
    # Inspect the resulting model
    model.summary()
    
    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #                                 monitor='val_accuracy',
    #                                 min_delta=0,
    #                                 patience=5,
    #                                 verbose=0,
    #                                 mode='auto',
    #                                 restore_best_weights=True,
    #                             )
    
    if type(trainlbs) == list:
        trainlbs = np.array(trainlbs)
    
    # Train the model! Takes about 20 sec/epoch
    batch_size = 16
    history = model.fit(trainimgs_res, trainlbs, 
                        validation_split = 0.25, 
                        epochs = ne, 
                        verbose = 1,
                        batch_size = batch_size,
                        shuffle=True,
                        class_weight = class_weights_dict,
                        #callbacks = [early_stopping]
                        )
    
    # Return the results!
    # On this model, we need to return the processed test images for validation 
    # in the later step
    return history, model, ne

def feature_extractor_fine_tuning(trainimgs, trainlbs, testimgs):
    """
    Building the Resnet50 model: a 256-dense NN is trained on ResNet50 features to classify the existance and location of instabilites and turbulence.
    After, the top layers of the CNN are unfrozen and allowed to vary. These are then further refined for our specific application.
    
    INPUTS
    ------------------
    trainimgs: (N, 224, 224, 3) numpy array of (224, 224, 3),
        image slices to train the model.
        
    trainlbs: (N,1) numpy array,
        binary classes
        
    testimgs: (M, 224, 224, 3) numpy array of (224, 224, 3),
        image slices to test the model.
    
    OUTPUTS: 
    ------------------
    history: keras NN model object,
        training history 
        
    model_FineTune: keras NN model object,
        Entire ResNet50-denseNN architecture with trained weights
        
    testimgs: (M, 224, 224, 3) numpy array of (224, 224, 3),
        image slices to test the model. Passed back to jive with some existing scripts better
        
    ne: int,
        number of epochs trained. Used for plotting later
    """
    # Form the base model
    base_model = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
    inputs = keras.Input(shape=(224,224,3))
    
    # Check length of model layers, if desired
    # print(len(base_model.layers))
    
    # Choose which layers to kept frozen or unfrozen
    for layer in base_model.layers[:155]: # the first 155 layers
        layer.trainable = False 
    
    # Construct the architecture
    x = inputs                                          # Start with image input
    x = base_model(x)                                   # pass thru Resnet50
    x = Flatten()(x)                                    # Flatten (just like above!)
    x = layers.Dense(128, activation = 'relu')(x)       # Pass thru the dense 256 arch
    x = layers.Dropout(0.5)(x)                          # Add dropout
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Final classification layer
    
    # Compile and train the model
    model_FineTune = Model(inputs, outputs)
    model_FineTune.summary()
    model_FineTune.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy']) # keep a low learning rate
    
    # Perform training. NOTE: takes around 4 min/epoch so be careful!
    ne = 20
    batch_size = 16
    history = model_FineTune.fit(trainimgs, trainlbs, 
                        validation_split = 0.25, 
                        epochs = ne, 
                        verbose = 1,
                        batch_size = batch_size,
                        shuffle=True)
    
    # Return the results!
    # On this model, we only need to return the testimages because we're NOT
    # running them thru the bottleneck first
    return history, model_FineTune, testimgs, ne


def plot_training_history(history, ne, name):
    """
    Plots the training performance vs epoch
    
    INPUTS
    ------------------
    history: keras NN model object,
        training history
        
    ne: int,
        number of epochs trained. Used for plotting later
        
    name: str,
        a string used to identify the plot. 'second-mode' or 'turbulence' recommended
    
    OUTPUTS: 
    ------------------
    None
    """
    
    epoch_list = list(range(1,ne + 1))
    # Making some plots to show our results
    f, (pl1, pl2) = plt.subplots(1, 2, figsize = (15,4), gridspec_kw = {'wspace': 0.3})
    t = f.suptitle('Neural Network Performance: ' + name, fontsize = 14)
    # Accuracy Plot
    pl1.plot(epoch_list, history.history['accuracy'], label = 'train accuracy')
    pl1.plot(epoch_list, history.history['val_accuracy'], label = 'validation accuracy')
    pl1.set_xticks(np.arange(0, ne + 1, 5))
    pl1.set_xlabel('Epoch')
    pl1.set_ylabel('Accuracy')
    pl1.set_title('Accuracy')
    leg1 = pl1.legend(loc = "best")
    # Loss plot for classification
    pl2.plot(epoch_list, history.history['loss'], label = 'train loss')
    pl2.plot(epoch_list, history.history['val_loss'], label = 'validation loss')
    pl2.set_xticks(np.arange(0, ne + 1, 5)) 
    pl2.set_xlabel('Epoch')
    pl2.set_ylabel('Loss')
    pl2.set_title('Classification Loss')
    leg2 = pl2.legend(loc = "best")
    plt.show()
    
    return 