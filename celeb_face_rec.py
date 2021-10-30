import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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

import time
import pafy

import socketserver
import requests
import io
from threading import Condition
import threading
from multiprocessing import Process, Value, Queue
from http import server

main_frame = None
lock = threading.Lock()

PAGE="""\
<html>
<head>
<title>Dobrinya - Smart Camera ver. 1.22474487139...</title>
</head>
<body>
<center><h1>Test camera</h1></center>
<center><img src="stream.mjpg" height=90% ></center>
</body>
</html>
"""

def web_set(img):
  global main_frame, lock
  with lock:
    main_frame = img.copy()

class StreamingHandler(server.BaseHTTPRequestHandler):
    global main_frame, lock
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
              while True:
                with lock:
                  frame = main_frame.copy()
                ret, jpeg = cv2.imencode('.jpg', frame)
                self.wfile.write(b'--FRAME\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', len(jpeg))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b'\r\n')
         
            except Exception as e:
                print(str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class Server():
  def __init__(self, ip='', port=2020):
    self.address = (ip, port)

  def start(self):    
    self.p = threading.Thread(target=self.process, args=()).start()

  def process(self):
    server = StreamingServer(self.address, StreamingHandler)
    server.serve_forever()


def HAAR(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(image, scaleFactor=1.1,
                                       minNeighbors=2, minSize=(30, 30),
                                       flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0: return None
    rects[:,2:] += rects[:,:2]
    boxes = []
    maximum = None
    for x1, y1, x2, y2 in rects:
      size = ((x2-x1) * (y2-y1))
      if maximum is None or maximum < size:
        maximum = size
        boxes.append([x1, y1, x2, y2])
    return boxes



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
 

if __name__ == '__main__':
##   mat = scipy.io.loadmat('imdb_crop/imdb.mat')
##
##   columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score", "celeb_names", "celeb_id"]
##   instances = mat['imdb'][0][0][0].shape[1]
##   df = pd.DataFrame(index = range(0,instances), columns = columns)
##
##   
##   for i in mat:
##       if i == "imdb":
##           current_array = mat[i][0][0]
##           for j in range(len(current_array)):
##               #print(j,". ",columns[j],": ",current_array[j][0])
##               df[columns[j]] = pd.DataFrame(current_array[j][0])
##
##   df = df[df['face_score'] != -np.inf]
##   df = df[df['second_face_score'].isna()]
##   df = df[df['face_score'] >= 3]
##
##   def extractNames(name):
##    return name[0]
##
##   df['celebrity_name'] = df['name'].apply(extractNames)
##
##   def getImagePixels(image_path):
##       return cv2.imread("imdb_crop/%s" % image_path[0])
##
##   df['pixels'] = df['full_path'].apply(getImagePixels)
##   df['face_vector_raw'] = df['pixels'].apply(findFaceRepresentation)   
##   df['similarity'] = df['face_vector_raw'].apply(findCosineSimilarity)
   
   model = loadVggFaceModel()
   #Pretrained weights: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
   from keras.models import model_from_json
   model.load_weights('vgg_face_weights.h5')
   

   youtube = not True

   if youtube:
     url = "https://www.youtube.com/watch?v=6BAAHW4w5DE"
     videoPafy = pafy.new(url)
     play = videoPafy.getbest(preftype="mp4")
   else:
     ip = "http://hackaton.sber-zvuk.com/hackathon_part_1.mp4"

   haar_path = "face.xml"
   resolution=(1920, 1080)

   if youtube:
     vid_capture = cv2.VideoCapture(play.url)
   else:
     vid_capture = cv2.VideoCapture(ip)

   s = Server()
   s.start()

   CODEC=cv2.VideoWriter_fourcc('M','J','P','G')
   vid_capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
   vid_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
   vid_capture.set(cv2.CAP_PROP_FOURCC, CODEC)
   #vid_capture.set(cv2.CAP_PROP_FPS, 60)
   vid_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

   while vid_capture.isOpened():
     # Obtain image
     ret, img = vid_capture.read()
     
     if not ret: continue
     h, w = img.shape[:2]

     # Resize
     #img = cv2.resize(img, (720,405))

     cascade = cv2.CascadeClassifier(haar_path)    
     boxes = HAAR(img)
     
     if boxes:
         for box in boxes:
            face = img[box[1]:box[3], box[0]:box[2]]
            # Blur
            if True:
              face = cv2.resize(face, (5, 5))
              face = cv2.resize(face, (box[3]-box[1],box[2]-box[0]))
            elif True:
              face = cv2.blur(face, (15, 15))

            img[box[1]:box[3], box[0]:box[2]] = face
            #cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), (255,255,255), thickness=2)
            face = cv2.resize(face, (224, 224))
            
            img_pixels = image.img_to_array(face)
            img_pixels = image.img_to_array(face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 127.5
            img_pixels -= 1
             
            yourself_representation = model.predict(img_pixels)[0,:]
            if min(yourself_representation) > 8:
               name = 'no body'
            else:
               name = np.argmin(yourself_representation)
            cv2.putText(img, str(name), (box[0], box[1]), 
                         cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
            print(name)
##            df['similarity'] = df['face_vector_raw'].apply(findCosineSimilarity)
##            df = df.sort_values(by=['similarity'], ascending=True)
##            instance = df.iloc[0]
##            name = instance['celebrity_name']
##
##            print(name)
            
     web_set(img)
     #cv2.imshow('feed', image)
     
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break

   vid_capture.release()
   

 



img = cv2.imread("sefik.jpg")
 
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
