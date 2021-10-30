import tensorflow as tf

import keras
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from keras.layers import Conv2D, AveragePooling2D
from keras.models import Model, Sequential
from keras import metrics
from keras.models import model_from_json

from sklearn.model_selection import train_test_split

import scipy.io
import numpy as np

import cv2

def loadVggFaceModel():
   model = Sequential()
   model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
   model.add(Convolution2D(64, (3, 3), activation='relu'))
   model.add(ZeroPadding2D((1,1)))
   model.add(Convolution2D(64, (3, 3), activation='relu'))
   model.add(MaxPooling2D((2,2), strides=(2,2)))
    
   model.add(ZeroPadding2D((1,1)))
   model.add(Convolution2D(128, (3, 3), activation='relu'))
   model.add(ZeroPadding2D((1,1)))
   model.add(Convolution2D(128, (3, 3), activation='relu'))
   model.add(MaxPooling2D((2,2), strides=(2,2)))
    
   model.add(ZeroPadding2D((1,1)))
   model.add(Convolution2D(256, (3, 3), activation='relu'))
   model.add(ZeroPadding2D((1,1)))
   model.add(Convolution2D(256, (3, 3), activation='relu'))
   model.add(ZeroPadding2D((1,1)))
   model.add(Convolution2D(256, (3, 3), activation='relu'))
   model.add(MaxPooling2D((2,2), strides=(2,2)))
    
   model.add(ZeroPadding2D((1,1)))
   model.add(Convolution2D(512, (3, 3), activation='relu'))
   model.add(ZeroPadding2D((1,1)))
   model.add(Convolution2D(512, (3, 3), activation='relu'))
   model.add(ZeroPadding2D((1,1)))
   model.add(Convolution2D(512, (3, 3), activation='relu'))
   model.add(MaxPooling2D((2,2), strides=(2,2)))
    
   model.add(ZeroPadding2D((1,1)))
   model.add(Convolution2D(512, (3, 3), activation='relu'))
   model.add(ZeroPadding2D((1,1)))
   model.add(Convolution2D(512, (3, 3), activation='relu'))
   model.add(ZeroPadding2D((1,1)))
   model.add(Convolution2D(512, (3, 3), activation='relu'))
   model.add(MaxPooling2D((2,2), strides=(2,2)))
    
   model.add(Convolution2D(4096, (7, 7), activation='relu'))
   model.add(Dropout(0.5))
   model.add(Convolution2D(4096, (1, 1), activation='relu'))
   model.add(Dropout(0.5))
   model.add(Convolution2D(2622, (1, 1)))
   model.add(Flatten())
   model.add(Activation('softmax'))
    
   vgg_face_descriptor = Model(inputs=model.layers[0].input, 
   outputs=model.layers[-2].output)
   return vgg_face_descriptor
 
def findFaceRepresentation(detected_face):
   try:
      detected_face = cv2.resize(detected_face, (224, 224))
       
      #normalize detected face in scale of -1, +1
      img_pixels = image.img_to_array(detected_face)
      img_pixels = np.expand_dims(img_pixels, axis = 0)
      img_pixels /= 127.5
      img_pixels -= 1
       
      representation = model.predict(img_pixels)[0,:]
   except:
      representation = None
 
   return representation
 

def findCosineSimilarity(source_representation, test_representation):
   try:
      a = np.matmul(np.transpose(source_representation), test_representation)
      b = np.sum(np.multiply(source_representation, source_representation))
      c = np.sum(np.multiply(test_representation, test_representation))
      return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
   except:
       return 10 #assign a large value in exception. similar faces will have small value.
 

model = loadVggFaceModel()
 
#Pretrained weights: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
from keras.models import model_from_json
model.load_weights('vgg_face_weights.h5')




img = cv2.imread("sefik.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(img, 1.3, 5)
 
for (x,y,w,h) in faces:
   detected_face = img[int(y):int(y+h), int(x):int(x+w)]
    
   try: #add 10% margin around the face
      margin = 10
      margin_x = int((w * margin)/100); margin_y = int((h * margin)/100)
      detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]
   except:
      print("detected face has no margin")
 
   detected_face = cv2.resize(detected_face, (224, 224))
 
#normalize in [-1, +1]
img_pixels = image.img_to_array(detected_face)
img_pixels = np.expand_dims(img_pixels, axis = 0)
img_pixels /= 127.5
img_pixels -= 1
 
yourself_representation = model.predict(img_pixels)[0,:]
