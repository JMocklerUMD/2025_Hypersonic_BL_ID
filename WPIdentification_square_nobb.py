# Version 2 of a CNN used to identify second-mode waves using ResNet50 for feature extraction

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from PIL import Image, ImageChops

import numpy as np
import os
import math
import random

import keras

import tensorflow as tf

from keras import optimizers, layers, regularizers

from keras.applications import ResNet50, vgg16

from keras.callbacks import EarlyStopping

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer

from keras.models import Model
from keras.models import Sequential

from keras.preprocessing import image

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

import matplotlib.pyplot as plt

import h5py

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import class_weight
#import cv2

from sklearn.utils.class_weight import compute_class_weight


#from tensorflow.keras.callbacks import EarlyStopping

#%% Function calls
'''
Function calls used throughout the script.
'''
def Shuffler(list1, list2):
	n1 = list1
	n2 = list2
	a = []
	for i in range(0,len(n1)):
		temp = [n1[i],n2[i]]
		a.append(temp)
	random.shuffle(a)
	n1new = []
	n2new = []
	for i in range(0,len(a)):
		n1new.append(a[i][0])
		n2new.append(a[i][1])

	return n1new, n2new

def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '>', printEnd = "\r"):
	percent = ("{0:." + str(decimals) + "f}").format(100*(iteration/float(total)))
	filledLength = int(length*iteration//total)
	bar = fill*filledLength + '-'*(length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
	if iteration == total:
		print()

# This function tells our feature extractor to do its thing
def get_bottleneck_features(model, input_imgs):
	print('Getting Feature Data From ResNet...')
	features = model.predict(input_imgs, verbose = 1)
	return features

#Old numpy array version of img_preprocess
def img_preprocess(input_image):
    input_image = np.stack((input_image,)*3,axis = -1)
    input_image = array_to_img(input_image)
    input_image = input_image.resize((224,224))
    input_image = img_to_array(input_image)
    input_image = input_image / 255.0
    return input_image

'''
#dataset version
def img_preprocess(image, label):
    image = tf.expand_dims(image, axis=-1) #change array shape for an image from [H,W] to [H,W,1] to conform to grayscale_to_rgb expectations
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, [224, 224])
    #image = tf.cast(image, tf.float32)
    #image = tf.keras.applications.resnet50.preprocess_input(image) #preprocesses x for resnet50 #seemed to make everything orange???
    #label = tf.cast(label, tf.float32) #or int32
    return image, label'''

#putting this before instead of imbedding within the ML model allows it to run parallel on CPU instead of GPU 
data_augmentation = keras.Sequential(
    
    )


#%% Read training data file
'''
Preprocessing: the following block of codes accept the image data from a 
big text file, parse them out, then process them into an array
that can be passed to the keras NN trainer
'''

print('Reading training data file')

# Write File Name
file_name = 'C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\wavepacket_labels_combined.txt'
if os.path.exists(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
else:
    raise ValueError("No training_data file detected")

lines_len = len(lines)
print(f"{lines_len} lines read")


#%% Write training data to required arrays and visualize
print('Begin writing training data to numpy array')

WP_io = []
#SM_bounds_Array = []
Imagelist = []
N_img, N_tot = 10, lines_len
i_sample, img_count = 0, 0
sampled_list = []

# Break when we aquire 100 images or when we run thru the 1000 frames
while (img_count < N_img) and (i_sample < N_tot):
    
    # Randomly sample image with probability N_img/N_tot
    # Skip image if in the 1-N_img/N_tot probability
    if np.random.random_sample(1)[0] < (1-N_img/N_tot):
        i_sample = i_sample + 1
        continue
    
    # Otherwise, we accept the image and continue with the processing
    curr_line = i_sample;
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
    full_image = np.array(image_data).astype(np.float32) #changed from float64
    full_image = full_image.reshape(image_size)  # Reshape to (rows, columns)
    
    #if full_image.shape != (64, 1280):
    #    print(f"Skipping image at line {i_sample+1} — unexpected size {full_image.shape}")
    #    continue
    
    slice_width = 64
    height, width = full_image.shape
    num_slices = width // slice_width
    
    # Only convert bounds to int if not sm_check.startswith('X')
    if not sm_check.startswith('X'):
        sm_bounds = list(map(int, sm_bounds))
        x_min, y_min, box_width, box_height = sm_bounds
        x_max = x_min + box_width
        y_max = y_min + box_height
    
    for i in range(num_slices):
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
    
    # Increment to the next sample image and image count
    i_sample = i_sample + 1
    img_count = img_count + 1
    
    # Inspect what images were selected later
    sampled_list.append(i_sample)

print('Done sampling images')

#%% Catches any arrays that are not correct size
#omit_array = []
#for i in range(len(Imagelist)):
#    if Imagelist[i].shape != (64, 64):
#        omit_array.append(i)

#Imagelist = [element for i, element in enumerate(Imagelist) if i not in omit_array]
#WP_io = [element for i, element in enumerate(WP_io) if i not in omit_array]


#%% Resizes the arrays
# Imagelist,WP_io = Shuffler(Imagelist, WP_io)
# Keras should shuffle our images for us - probably don't need to do!
Imagelist = np.array(Imagelist)
Imagelist = np.array([img_preprocess(img) for img in Imagelist])
WP_io = np.array(WP_io)
print("Done inputting to np.array's")

#%%fake troubleshooting data
'''
num_samples = 32
image_shape = (64, 64)
x = np.random.rand(num_samples, *image_shape).astype(np.float32)  # Already normalized
y = np.random.randint(0, 2, size=(num_samples,)).astype(np.int32)

'''
#%% Create dataset - see https://www.youtube.com/watch?v=OqWbsbLhKws&list=WL

# 2. Shuffle entire dataset before splitting
# -------------------------------
indices = np.random.permutation(len(Imagelist))
images = Imagelist[indices]
labels = WP_io[indices]

# -------------------------------
# 3. Split into train, val, test (60/20/20)
# -------------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)
# Now: 60% train, 20% val, 20% test

print(type(X_train), type(y_train))
print(X_train.shape, y_train.shape)
print(X_train.dtype, y_train.dtype)

classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train)

# Convert to dictionary format required by Keras
class_weights_dict = dict(zip(classes, class_weights))
print("Class weights:", class_weights_dict)

# -------------------------------
# 5. Create tf.data pipelines with .map()
# -------------------------------
def make_dataset(X, y, batch_size=16, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    #ds = ds.map(img_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds



batch_size = 16
#ds = make_dataset(x,y,batch_size)

train_ds = make_dataset(X_train, y_train, batch_size)
val_ds = make_dataset(X_val, y_val, batch_size, shuffle=False)
test_ds = make_dataset(X_test, y_test, batch_size, shuffle=False)

for image_batch, label_batch in train_ds.take(1):
    image = image_batch[0]  # shape: (128, 128, 3)
    plt.imshow(image.numpy())  # Convert tensor to NumPy for matplotlib
    plt.title(f"Label: {label_batch.numpy()[0]}")
    plt.axis("off")
    plt.show()
    
for image, label in train_ds.take(1):
    print("Image shape:", image.shape)
    print("Image min/max:", tf.reduce_min(image), tf.reduce_max(image))
    print("Label:", label)

print('Done making datasets')
#%% Create model

conv_base = ResNet50(weights="imagenet", 
                     include_top=False,
                     input_shape=(224,224,3))
conv_base.trainable = False;

model = tf.keras.Sequential([
    #conv_base,
    
    #test without conv_base
    tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(),

    
    tf.keras.layers.Flatten(), #.GlobalAveragePooling2D() could be more efficient and only marginally less accurate
    tf.keras.layers.Dense(256, 
                     activation='relu'
                     #kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4),     # Regularization penality term
                     #bias_regularizer=regularizers.L2(1e-4)
                     ),  # Additional regularization penalty term
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4), #was 1e-6 #maybe increase to 1e-4 or even 1e-3
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])


#early stopping code from Google search AI
early_stopping = EarlyStopping(
        monitor='val_accuracy',  # Metric to monitor (e.g., validation loss)
        patience=5,        # Number of epochs with no improvement to wait
        min_delta=0.001,    # Minimum change to be considered an improvement
        restore_best_weights=True # Whether to restore the model weights to the best epoch
    )

#model.summary()

single_batch = train_ds.take(1).repeat()
    
    # Train the model! Takes about 20 sec/epoch
ne = 20
batch_size = 16
history = model.fit(single_batch, 
                        validation_data = val_ds, 
                        epochs = ne, 
                        verbose = 1,
                        steps_per_epoch=10, #single_batch intential overfit troubleshooting
                        #batch_size = batch_size,
                        shuffle=True,
                        class_weight = class_weights_dict,
                        callbacks=[early_stopping]
                        )

# -------------------------------
# 8. Evaluate on test set
# -------------------------------
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")


#%% Perform the visualization

#Visualization: inspect how the training went

rne = len(history.history['accuracy'])

#model.save('ClassifierV1m.h5')
epoch_list = list(range(1,rne + 1))
# Making some plots to show our results
f, (pl1, pl2) = plt.subplots(1, 2, figsize = (15,4), gridspec_kw = {'wspace': 0.3})
t = f.suptitle('Neural Network Performance', fontsize = 14)
# Accuracy Plot
pl1.plot(epoch_list, history.history['accuracy'], label = 'train accuracy')
pl1.plot(epoch_list, history.history['val_accuracy'], label = 'validation accuracy')
pl1.set_xticks(np.arange(0, rne + 1, 5))
pl1.set_xlabel('Epoch')
pl1.set_ylabel('Accuracy')
pl1.set_title('Accuracy')
leg1 = pl1.legend(loc = "best")
# Loss plot for classification
pl2.plot(epoch_list, history.history['loss'], label = 'train loss')
pl2.plot(epoch_list, history.history['val_loss'], label = 'validation loss')
pl2.set_xticks(np.arange(0, rne + 1, 5)) 
pl2.set_xlabel('Epoch')
pl2.set_ylabel('Loss')
pl2.set_title('Classification Loss')
leg2 = pl2.legend(loc = "best")
plt.show()


#%%old stuff
'''
batch_size = 16

#dataset = dataset.shuffle(5) #ignore first # from bummer, then take the next available; higher buffer # is more random; ideal buffer is equal to the size of the dataset, but that can be unrealistically large
#1024 COULD BE choosen due to its use as an example in the above YT video - small enough to be reasonable, large enough to be somewhat random; if len(dataset)<1024, it shuffles as if buffer==len(dataset)
shuffled_dataset = dataset.shuffle(lines_len, reshuffle_each_iteration=False)

length = sum(1 for _ in shuffled_dataset)

indexed_dataset = shuffled_dataset.enumerate()

train_size = int(0.6 * length)
val_size = int(0.2 * length)

train_dataset = indexed_dataset.filter(lambda i, _: i < train_size)
val_dataset = indexed_dataset.filter(lambda i, _: (i >= train_size) & (i < train_size + val_size))
test_dataset = indexed_dataset.filter(lambda i, _: i >= train_size + val_size)

'''
'''
print(lines_len)
print(length)
print("Train:", sum(1 for _ in train_dataset))
print(train_size)
print("Val:", sum(1 for _ in val_dataset))
print(val_size)
print("Test:", sum(1 for _ in test_dataset))
'''
'''

# Remove index and preprocess
train_dataset = train_dataset.map(lambda i, x: x).map(img_preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(lambda i, x: x).map(img_preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(lambda i, x: x).map(img_preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print('90% done creating datasets - everything, but repeating')
#%% Get unique classes and compute weights
labels = []
for _, label in train_dataset:
    labels.extend(label.numpy())
labels = np.array(labels)
classes = np.unique(labels)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=labels)

# Convert to dictionary format required by Keras
class_weights_dict = dict(zip(classes, class_weights))
print("Class weights:", class_weights_dict)

#%% allow train and val datasets to repeat (to avoid running out of dataset batches)
print("Validation samples:", sum(1 for _ in val_dataset))
train_dataset = train_dataset.repeat()
val_dataset = val_dataset.repeat()

print('Done creating datasets')
#%% Split the test and train images

#trainimgs, testimgs, trainlbs, testlbls = train_test_split(Imagelist_resized,WP_io, test_size=0.2, random_state=69)
#print("Done Splitting")

#%% create the model - from YT Video "Transfer Learning with CNNs - Deep Learning with Tensorflow | Ep. 20" by Kody Simpson
conv_base = ResNet50(weights="imagenet", include_top=False,input_shape=(224,224,3))
conv_base.trainable = False;

inputs = keras.Input(shape=(224,224,3))
x = conv_base(inputs, training=False) #base of resnet50
x = layers.Flatten()(x)
x = layers.Dense(256, 
                 activation='relu',
                 kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4),     # Regularization penality term
                 bias_regularizer=regularizers.L2(1e-4)  # Additional regularization penalty term
                 )(x) 
x = layers.Dropout(0.5)(x) # Add dropout to make the system more robust
outputs = layers.Dense(1, activation = 'sigmoid')(x) # final classification layer

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-6), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

model.summary()

print('Done compiling model')

#%%some troubleshooting
#import matplotlib.pyplot as plt

# Take one batch from the dataset
for x_batch, y_batch in train_dataset.take(1):
    print("Batch shape:", x_batch.shape)

    # Loop through the first N samples in the batch
    N = min(5, x_batch.shape[0])  # Adjust how many to visualize

    for i in range(N):
        img = tf.keras.utils.array_to_img(x_batch[i])
        label = y_batch[i].numpy()

        plt.figure()
        plt.title(f"Label: {label}")
        plt.imshow(img)
        plt.axis('off')
        plt.show()
'''

#%%Train model
'''

ne = 5
#batch_size = 16
history = model.fit(train_dataset, 
                     validation_data = val_dataset, 
                     validation_steps = int(val_size/batch_size),
                     epochs = ne, 
                     steps_per_epoch = int(train_size/batch_size), #how many batches per epoch to go through, for None (I think) it tries to divide them evenly
                     verbose = 1,
                     shuffle=True, #ignored since our input is a tf.data.Dataset
                     #batch_size = batch_size,
                     class_weight = class_weights_dict
                     #,callbacks=[early_stopping]
                     )
print('Done training model')'''
#%% Train the feature extractor model only
'''
def feature_extractor_training(train_dataset, test_dataset):
    """
    Building the Resnet50 model: images are first passed through the Reset50 model
    prior to passing through one last NN layer that we will define. Initialize
    this code block once!
    """

    # Bringing in ResNet50 to use as our feature extractor
    model1 = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
    output = model1.output
    output = tf.keras.layers.Flatten()(output)
    resnet_model = Model(model1.input,output)

    # Locking in the weights of the feature detection layers
    resnet_model.trainable = False
    for layer in resnet_model.layers:
    	layer.trainable = False
    
    
    
    #Defining and training our classification NN: after passing through resnet50,
    #images are then passed through this network and classified. 
    
    #trainimgs_res = get_bottleneck_features(resnet_model, train_dataset.Imagelist)
    testimgs_res = get_bottleneck_features(resnet_model, test_dataset.Imagelist)
    
    # Generate an input shape for our classification layers
    input_shape = resnet_model.output_shape[1]
    
    # Get unique classes and compute weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_dataset.WP_io),
        y=train_dataset.WP_io
        )
    
    # Convert to dictionary format required by Keras
    class_weights_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weights_dict)
    
    # Added the classification layers
    model = Sequential()
    model.add(InputLayer(input_shape = (input_shape,)))
    model.add(Dense(256,                                        # NN dimension            
                    activation = 'relu',                        # Activation function at each node
                    input_dim = input_shape,                    # Input controlled by feature vect from ResNet50
                    kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4),     # Regularization penality term
                    bias_regularizer=regularizers.L2(1e-4)))                    # Additional regularization penalty term
    
    model.add(Dropout(0.5))     # Add dropout to make the system more robust
    model.add(Dense(1, activation = 'sigmoid'))     # Add final classification layer
    
    # Compile the NN
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-6), 
                  loss = 'binary_crossentropy', 
                  metrics = ['accuracy'])
    
    # Inspect the resulting model
    model.summary()
    
    
    #early stopping code from Google search AI
    early_stopping = EarlyStopping(
        monitor='val_accuracy',  # Metric to monitor (e.g., validation loss)
        patience=5,        # Number of epochs with no improvement to wait
        min_delta=0.001,    # Minimum change to be considered an improvement
        restore_best_weights=True # Whether to restore the model weights to the best epoch
    )
    
    # Train the model! Takes about 20 sec/epoch
    ne = 20
    batch_size = 16
    history = model.fit(train_dataset, 
                        validation_split = 0.25, 
                        epochs = ne, 
                        verbose = 1,
                        batch_size = batch_size,
                        shuffle=True,
                        class_weight = class_weights_dict
                        #,callbacks=[early_stopping]
                        )
    
    # Return the results!
    # On this model, we need to return the processed test images for validation 
    # in the later step
    return history, model, testimgs_res, ne
'''
#%% Train the fine tuning model

'''

def feature_extractor_fine_tuning(trainimgs, trainlbs, testimgs):
    
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
    x = layers.Dense(256, activation = 'relu')(x)       # Pass thru the dense 256 arch
    x = layers.Dropout(0.5)(x)                          # Add dropout
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Final classification layer
    
    # Compile and train the model
    model_FineTune = Model(inputs, outputs)
    model_FineTune.summary()
    model_FineTune.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-6), 
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

'''

#%% Call fcn to train the model!
'''
history, model, testimgs_res, ne = feature_extractor_training(train_dataset, test_dataset)
#history, model, testimgs_res, ne = feature_extractor_fine_tuning(trainimgs, trainlbs, testimgs)
print("Training Complete!")'''

#%% Perform the visualization
'''
#Visualization: inspect how the training went

#model.save('ClassifierV1m.h5')
epoch_list = list(range(1,ne + 1))
# Making some plots to show our results
f, (pl1, pl2) = plt.subplots(1, 2, figsize = (15,4), gridspec_kw = {'wspace': 0.3})
t = f.suptitle('Neural Network Performance', fontsize = 14)
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
'''
#%% Implement some statistics
'''
# Check how well we did on the test data!
test_res= model.predict(testimgs_res)
test_res_binary = np.round(test_res)

# build out the components of a confusion matrix
n00, n01, n10, n11 = 0, 0, 0, 0 

for i, label_true in enumerate(test_dataset.WP_io):
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
    
acc = (n00 + n11) / len(test_dataset.WP_io) # complete accuracy
Se = n11 / n1 # true positive success rate, recall
Sp = n00 / n0 # true negative success rate
Pp = n11 / (n11 + n01) # correct positive cases over all pred positive
Np = n00 / (n00 + n10) # correct negative cases over all pred negative
Recall = TP/(TP+FN) # Probability of detection
FRP = FP/(FP+TN) # False positive, probability of a false alarm

# Rate comapared to guessing
# MICE -> 1: perfect classification. -> 0: just guessing
A0 = (n0/len(test_dataset.WP_io))**2 + (n1/len(test_dataset.WP_io))**2
MICE = (acc - A0)/(1-A0)   

#%% Print out the summary statistics
ntot = len(test_dataset.WP_io)
print("------------Test Results------------")
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
'''

#%% Save off the model, if desired
#model.save('C:\\Users\\tyler\\Desktop\\NSSSIP25\\CROPPEDrun33\\Test1\\run33\\run33_relabeled.keras')

