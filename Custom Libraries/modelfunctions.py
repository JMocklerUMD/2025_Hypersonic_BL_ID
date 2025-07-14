from keras import regularizers
from keras.applications import resnet50
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, InputLayer
from keras.models import Model, Sequential

import tensorflow as tf

from sklearn.utils import class_weight

import numpy as np



# This function tells our feature extractor to do its thing
def get_bottleneck_features(model, input_imgs, verbose = 0): #not verbose by default
    '''
    Retrives the ResNet50 feature vector
    INPUTS:  model:      resnet50 Keras model
             input_imgs: (N, 224, 224, 3) numpy array of (224, 224, 3) images to extract features from       
    OUTPUTS: featues:   (N, 100352) numpy array of extracted ResNet50 features
    '''

    if input_imgs.shape == (224,224,3): #adds batch dimension for single images
        input_imgs = np.expand_dims(input_imgs, axis=0)                # Shape: (1, 224, 224, 3)
        if verbose == 1:
            print('note: batch size = 1')
    if verbose == 1:
        print('Getting Feature Data From ResNet...')
    features = model.predict(input_imgs, verbose = verbose)
    return features

def resnet_feature_extractor():
    
    # Bringing in ResNet50 to use as our feature extractor
    model1 = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
    output = model1.output
    output = tf.keras.layers.Flatten()(output)
    resnet_model = Model(model1.input,output)

    # Locking in the weights of the feature detection layers
    resnet_model.trainable = False
    for layer in resnet_model.layers:
    	layer.trainable = False
        
    return resnet_model

def feature_extractor_training(trainimgs, trainlbs, testimgs, ne):
    """
    Building the Resnet50 model: a 256-dense NN is trained on ResNet50 features to classify the images.
    
    INPUTS: trainimgs_res:  (N, 100532) numpy array of feature vectors for each image slice to train the model.
            trainlbs:       (N,1) numpy array of binary classes
            testimgs_res:   (M, 100532) numpy array of feature vectors for each image slice to test the model.
    
    OUTPUTS: history:       keras NN model training history object
             model:         trained NN model of JUST the 256 dense NN
             testimgs_res:  (M, 100532) ResNet50 feature vector for each test image slice
             ne:            number of epochs trained
    """

    resnet_model = resnet_feature_extractor()
    
    #Defining and training our classification NN: after passing through resnet50,
    #images are then passed through this network and classified. 
    trainimgs_res = get_bottleneck_features(resnet_model, trainimgs, verbose=1)
    testimgs_res = get_bottleneck_features(resnet_model, testimgs, verbose=1)
    
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
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
                                    monitor='val_accuracy',
                                    min_delta=0,
                                    patience=5,
                                    verbose=0,
                                    mode='auto',
                                    restore_best_weights=True,
                                )
    
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
    return history, model, testimgs_res, ne