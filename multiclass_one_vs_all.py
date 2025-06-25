#%% Import libraries
import numpy as np
import os
import math
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import tensorflow as tf

import keras
from keras import optimizers, layers, regularizers
from keras.applications import resnet50
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

#%% Input files and run settings

N_positive_classes = 2

if N_positive_classes >= 1:
    file_name1 = ''
    N_imgs_1 = 200
if N_positive_classes >= 2:
    file_name2 = ''
    N_imgs_2 = 200
if N_positive_classes >= 3:
    file_name3 = ''
    N_imgs_3 = 200
if N_positive_classes >= 4:
    file_name4 = ''
    N_imgs_4 = 200
if N_positive_classes >= 5:
    file_name5 = ''
    N_imgs_5 = 200
if N_positive_classes >= 6:
    file_name6 = ''
    N_imgs_6 = 200

ne = 20             # Number of epoches
early_stop = False  # Do early stopping?

whole_set_file_name = file_name1
plot_flag = 1       # View the images? MUCH SLOWER (view - 1, no images - 0)
N_frames = -1       # Number of frames to go through for whole-set
                    # If you want the whole-set -> N_frames = -1

if N_positive_classes < 1 and  N_positive_classes > 6 and not isinstance(N_positive_classes,int):
    raise ValueError('N_positive_classes must be an integer 1 to 6')

#%% Define functions

