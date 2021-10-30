#pip install pafy
#pip install youtube_dl
import os
import cv2
import numpy as np
import time
import pickle
import socketserver
import requests
import imutils
import io
from threading import Condition
import threading
from multiprocessing import Process, Value, Queue
from http import server
import pafy
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())


main_frame = None
lock = threading.Lock()

PAGE="""\
<html>
<head>
<title>Dobrinya - Smart Camera ver. 1.22474487139...</title>
</head>
<body>
<center><h1>Test camera</h1></center>
<center><img src="stream.mjpg" width=100% ></center>
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

if __name__ == '__main__':
  youtube = True

  if youtube:
    url = "https://www.youtube.com/watch?v=e-ORhEE9VVg"
    videoPafy = pafy.new(url)
    play = videoPafy.getbest(preftype="mp4")
  else:
    ip = "http://hackaton.sber-zvuk.com/hackathon_part_1.mp4"

  haar_path = "face.xml"

  if youtube:
   vid_capture = cv2.VideoCapture(play.url)
  else:
   vid_capture = cv2.VideoCapture(ip)

  s = Server()
  s.start()

  CODEC=cv2.VideoWriter_fourcc('M','J','P','G')
  vid_capture.set(cv2.CAP_PROP_FOURCC, CODEC)
  vid_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

  print('Go')
  
  while vid_capture.isOpened():
    ret, frame = vid_capture.read()

    if not ret:
      continue

    (h0, w0) = frame.shape[:2]
    img = frame.copy()
##    img = imutils.resize(img, width=600)
    (h, w) = img.shape[:2]
    k = 1

    imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(img, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
      confidence = detections[0, 0, i, 2]
      if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = frame[startY*k:endY*k, startX*k:endX*k]
        (fH, fW) = face.shape[:2]

        if fW < 20 or fH < 20:
          continue

        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                         (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()
        preds = recognizer.predict_proba(vec)[0]

        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        if proba > 0.7:
          text = "{}: {:.2f}%".format(name, proba * 100)
          y = startY*k - 10 if startY*k - 10 > 10 else startY*k + 10

          cv2.rectangle(frame, (startX*k, startY*k), (endX*k, endY*k),
                        (0, 0, 255), 2)
          cv2.putText(frame, text, (startX*k, y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

          face = cv2.resize(face, (5, 5))
          face = cv2.resize(face, (fW, fH))

          frame[startY*k:endY*k, startX*k:endX*k] = face

        web_set(frame)
 
  vid_capture.release()
