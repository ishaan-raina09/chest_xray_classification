#!/usr/bin/env python
# coding: utf-8

input tensorflow

import tensorflow

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input

import os
import matplotlib.pyplot as plt
import numpy as np

# conda install scikit-learn
from sklearn import metrics
import time

ls

#Define constants
img_width, img_height = 150, 150

train_data_dir = r"D:\chest_xray\train"
test_data_dir =  r"D:\chest_xray\test"

nb_train_samples = 1341+3875   

validation_data_dir = r"D:\chest_xray\val"

nb_val_samples = 16   # Actual: 400 + 400 (more) =  800

# Some hyperparameters
batch_size = 32          
epochs = 5              
test_generator_samples = 390+234
test_batch_size = 16   
input_shape = (img_width, img_height, 3)

model = Sequential()

model.add(Input(shape = input_shape ))

#Add Conv2D layer
model.add(Conv2D(
filters=45,   
    #total bias = 32
    kernel_size=(3, 3),        
    strides = (1,1),          
use_bias=True,             
padding='valid',           
name="1st_conv_layer"
)
         )

model.add(Activation('relu'))  

model.summary()

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32,
                (3, 3),
                activation = 'relu',
                name = "2nd_con_layer"))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), name = "3rd_conv_layer"))

model.summary()

model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten(name = "FlattenedLayer"))
model.summary()

#Classification Layer
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid')) 

model.compile(
              loss='binary_crossentropy',  
              optimizer='rmsprop',         
              metrics=['accuracy'])  

def preprocess(img):
    return img


tr_dtgen = ImageDataGenerator(
                              rescale=1. / 255,      
                              shear_range=0.2,       
                              zoom_range=0.2,
                              horizontal_flip=True,
                              preprocessing_function=preprocess
                              )

train_generator = tr_dtgen.flow_from_directory(
                                               train_data_dir,       
                                               target_size=(img_width, img_height),  
                                               batch_size=batch_size,  
                                               class_mode='binary'   
                                                )

val_dtgen = ImageDataGenerator(rescale=1. / 255)

validation_generator = val_dtgen.flow_from_directory(
                                                     validation_data_dir,
                                                     target_size=(img_width, img_height),   # Resize images
                                                     batch_size=batch_size,    # batch size to augment at a time
                                                     class_mode='binary'  # Return 1D array of class labels
                                                     )

start = time.time()
history = model.fit_generator(
                              generator = train_generator,
                              
                              steps_per_epoch=nb_train_samples // batch_size,
                              # No of epochs
                              epochs=epochs,
                              #validation data from validation generator
                              validation_data=validation_generator,
                              verbose = 1,
                              validation_steps=nb_validation_samples // batch_size
                              )
