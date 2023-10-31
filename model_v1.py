import os
from os import listdir
#import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization,Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools

import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



train_path =r"Data Set/train"
test_path =r"Data Set/test"
valid_path =r"Data Set/valid"

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=train_path,target_size=(224,224),classes=['cats','dogs'],batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=valid_path,target_size=(224,224),classes=['cats','dogs'],batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=test_path,target_size=(224,224),classes=['cats','dogs'],batch_size=10,shuffle=False)

imgs, labels = next(train_batches)

def plotimgs(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes= axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotimgs(imgs)
#print(labels)

model = Sequential([Conv2D(filters = 32, kernel_size=(3,3), activation='relu', padding = 'same', input_shape=(224, 224,3)),
MaxPool2D(pool_size=(2,2),strides=2),
Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),
MaxPool2D(pool_size=(2,2),strides=2),
Flatten(),
Dense(units=2, activation='softmax'),])

model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)