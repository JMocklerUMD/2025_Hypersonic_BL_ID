# Version 2 of a CNN used to identify second-mode waves using ResNet50 for feature extraction

"""
Importing a bunch of crap
"""

# Gotta make sure we're using Python 3 first
import sys
if sys.version_info[0] < 3:
	raise Exception("Python 3 not being used!")

import os

import keras
from keras import optimizers

from keras.applications import resnet50, vgg16

from keras.callbacks import EarlyStopping

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer

from keras.models import Model
from keras.models import Sequential

from keras.preprocessing import image

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

import matplotlib.pyplot as plt

# import h5py

import numpy as np

import opencv as cv2

import random

"""
Progress bar function for use in future functions
"""

def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '>', printEnd = "\r"):
	percent = ("{0:." + str(decimals) + "f}").format(100*(iteration/float(total)))
	filledLength = int(length*iteration//total)
	bar = fill*filledLength + '-'*(length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
	if iteration == total:
		print()

"""
Shuffler function to randomize the order of the incoming data
"""

def Shuffler(list1, list2, list3):
	n1 = list1
	n2 = list2
	n3 = list3
	a = []
	for i in range(0,len(n1)):
		temp = [n1[i],n2[i],n3[i]]
		a.append(temp)
	random.shuffle(a)
	n1new = []
	n2new = []
	n3new = []
	for i in range(0,len(a)):
		n1new.append(a[i][0])
		n2new.append(a[i][1])
		n3new.append(a[i][2])

	return n1new, n2new, n3new

"""
Function for reading in image data
"""

def read_image_data(folder, filename):
	filelist = folder
	namelist = []
	imagelist = []
	labellist = []
	boxcoordlist = []
	i = 0
	m = 0
	q = 0
	for file in os.listdir(filelist):
		namelist.append(os.path.join(filelist, file))
	keyword = 'Not'
	namelist.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))
	progress_bar(0, len(namelist), prefix = 'Loading Images:', suffix = 'Complete', length = 50)
	for image_name in namelist:
		#print(image_name)
		img = load_img(image_name, target_size = (224,224))
		img = img_to_array(img)
		imagelist.append(img)
		if keyword in image_name:
			labellist.append(1)
			q = q + 1
		else:
			labellist.append(0)
		i = i + 1
		progress_bar(i, len(namelist), prefix = 'Loading Images:', suffix = 'Complete', length = 50)
	progress_bar(0, len(namelist), prefix = 'Loading Bounding Box Coordinates:', suffix = 'Complete', length = 50)
	boxfile = open(filename, 'r')
	for j in range(0,len(namelist)):
		boxcoord = (boxfile.readline())
		boxcoord = list(boxcoord.split(' '))
		for k in range(0,len(boxcoord)):
			boxcoord[k] = int(boxcoord[k])
		boxcoordlist.append(boxcoord)
		m = m+1
		progress_bar(m, len(namelist), prefix = 'Loading Bounding Box Coordinates:', suffix = 'Complete', length = 50)
	boxfile.close
	imagelist, labellist, boxcoordlist = Shuffler(imagelist, labellist, boxcoordlist)
	imagelist = np.array(imagelist)
	boxcoordlist = np.array(boxcoordlist)
	print(q)

	return imagelist, labellist, boxcoordlist

"""
Building the model below
"""

# Bringing in ResNet50 to use as our feature extractor
model1 = resnet50.ResNet50(include_top = False, weights ='imagenet', input_shape = (224,224,3))
output = model1.output
output = keras.layers.Flatten()(output)
resnet_model = Model(model1.input,output)

# Locking in the weights of the feature detection layers
resnet_model.trainable = False
for layer in resnet_model.layers:
	layer.trainable = False

# This function tells our feature extractor to do its thing
def get_bottleneck_features(model, input_imgs):
	print('Getting Feature Data From ResNet...')
	features = model.predict(input_imgs, verbose = 1)
	return features

# Generate an input shape for our classification layers
input_shape = resnet_model.output_shape[1]

# Now we'll add new classification layers
model = Sequential()
model.add(InputLayer(input_shape = (input_shape,)))
model.add(Dense(64, activation = 'relu', input_dim = input_shape))
model.add(Dropout(0.3))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))

# Compiling our masterpiece
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# model.summary()

# Second set of layers for the bounding box coordinates
model2 = Sequential()
model2.add(InputLayer(input_shape = (input_shape,)))
model2.add(Dense(64, activation = 'relu', input_dim = input_shape))
model2.add(Dropout(0.3))
model2.add(Dense(64, activation = 'relu'))
model2.add(Dropout(0.3))
model2.add(Dense(4, activation = 'linear'))

# Compiling our second masterpiece
model2.compile(optimizer = 'rmsprop', loss = 'mse')

# Implementing an early stopping monitor (optional for now)
early_stopping_monitor = EarlyStopping(patience = 3)

"""
Let's train the model and take a look at how it does
"""

# Running the functions to bring in our images and labels
trainimgs, trainlbls, boxcoords = read_image_data('Processed_Main_Test_Mean','Processed_Main_Test_Bounding_Boxes.txt')
trainimgs_res = get_bottleneck_features(resnet_model, trainimgs)

# Number of Epochs to Train on:
ne = 30

# Training the classification model and checking accuracy
history = model.fit(trainimgs_res, trainlbls, validation_split = 0.2, epochs = ne, verbose = 1)

# Doing the same with the bounding box model
history2 = model2.fit(trainimgs_res, boxcoords, validation_split = 0.2, epochs = ne, verbose = 1)

# Generating a range of epochs run
epoch_list = list(range(1,ne + 1))

model.save('ClassifierV1m.h5')
model2.save('RegressorV1m.h5')

# Making some plots to show our results
f, (pl1, pl2, pl3) = plt.subplots(1, 3, figsize = (15,4), gridspec_kw = {'wspace': 0.3})
t = f.suptitle('Neural Network Performance', fontsize = 14)
# Accuracy Plot
pl1.plot(epoch_list, history.history['acc'], label = 'train accuracy')
pl1.plot(epoch_list, history.history['val_acc'], label = 'validation accuracy')
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
# Loss plot for bounding boxes
pl3.plot(epoch_list, history2.history['loss'], label = 'train loss')
pl3.plot(epoch_list, history2.history['val_loss'], label = 'validation loss')
pl3.set_xticks(np.arange(0, ne + 1, 5))
pl3.set_xlabel('Epoch')
pl3.set_ylabel('Loss')
pl3.set_title('Regression Loss')
leg3 = pl3.legend(loc = "best")
plt.show()

# Displaying a sample image with boxes drawn on it
# demoimg = cv2.imread('Processed_4017\\Processed_SMW_Present_2.tif')
# demobox = boxcoords[1]
# cv2.rectangle(demoimg,(demobox[0],demobox[1]),(demobox[2]+demobox[0], demobox[3]+demobox[1]),(0,255,0),2)
# cv2.rectangle(demoimg,(demobox[0],demobox[1]),(demobox[2]+demobox[0], demobox[3]+demobox[1]),(255,0,0),2)
# cv2.imshow('test',demoimg)
# cv2.waitKey()

# Passing new images to the network for predictions
predimgs, predlbls, inputboxcoords = read_image_data('Processed_Predictions','Processed_Predictions_Bounding_Boxes.txt')
predimgs_res = get_bottleneck_features(resnet_model, predimgs)
score = model.evaluate(predimgs_res, predlbls, verbose = 1)
print(score[1]*100, '\n')
predboxcoords = model2.predict(predimgs_res)
img = array_to_img(predimgs[1])
for i in range(0,len(predlbls)):
	inputbox = inputboxcoords[i]
	outputbox = predboxcoords[i]
	image = cv2.imread('Processed_Predictions\\Processed_SMW_Present_2001.tif')
	cv2.rectangle(image,(inputbox[0],inputbox[1]),(inputbox[2]+inputbox[0], inputbox[3]+inputbox[1]),(0,255,0),2)
	cv2.rectangle(image,(outputbox[0],outputbox[1]),(outputbox[2]+outputbox[0], outputbox[3]+outputbox[1]),(255,0,0),2)
	cv2.imshow('test', image)
	cv2.waitKey()
