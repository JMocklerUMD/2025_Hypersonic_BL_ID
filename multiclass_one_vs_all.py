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

file_name1 = ''
N_imgs_1 = 200

ne = 20

