
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import basic essential libraries 
import matplotlib.pyplot as plt
import path
import os
%matplotlib inline

# import  keras libraries to build model and conv net
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten
from pathlib import Path

# current dataset only has 4k images
# so image agumentation to virtually increase size of dataset 
# help (ImageDataGenerator?)

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

data_agumentation=ImageDataGenerator(rescale=0./255,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True

)

# load training data
train_data=data_agumentation.flow_from_directory(directory='../enter relative path of the training directory/',
                                                 target_size=(256,256),
                                                 class_mode='categorical',
                                                batch_size=32 
                                               )
#load validation data
val_data=ImageDataGenerator().flow_from_directory(directory='../enter relative path of the validation directory/',
                                                      target_size=(256,256),
                                                       class_mode='categorical'
                                                      )                                               

# Model 

#Early stopping and callback

# import modelcheckpoint and earlystopping  

from tensorflow.python.keras.callbacks import ModelCheckpoint,EarlyStopping

# checkpoint monitors  the given parameter and save the model automatically
# here given parameter to monitor is val_loss
# it moniters the  val loss of each epoch and val_loss is lower than previous one it save the current model ad wieght

checkpoint=ModelCheckpoint("currency_detector_smal_model.h5", monitor='val_loss',verbose=1, save_best_only=True,save_weights_only=False,mode='auto', period=1)

# early stopping .. it stops the trainng phase if there is no improvement in the model
# patience defines how many epoch can b ignored before forcefully stoping the model

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')


#define model add layers and compile

define model add layers and compile

# compile the model with adam optimizer, categorical_croosentropy loss function
my_new_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Fit the model with train and val dataset

# fit the model with train data and validation data 
# epoch 50
my_new_model.fit_generator(
        train_data,
        epochs = 50,
        validation_data=val_data,
        callbacks=[checkpoint,early])

#save model into JSON

# save the json model  

model_json = my_new_model.to_json()
with open("resnet_50_model.json", "w") as json_file:
    json_file.write(model_json)

