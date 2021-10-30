#pip install pafy
#pip install youtube_dl

# cd /d D:\hakaton\SberZvuk-Tech-Days
# activate exercise_control
# python web_face_mp.py

import cv2
import numpy as np
import time

import mediapipe as mp
import socketserver
import requests
import io
from threading import Condition
import threading
from multiprocessing import Process, Value, Queue
from http import server

main_frame = None
lock = threading.Lock()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

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

if __name__ == '__main__':
    
    ip = "http://hackaton.sber-zvuk.com/hackathon_part_1.mp4"
    resolution=(1920, 1080)

    vid_capture = cv2.VideoCapture(ip)

    s = Server()
    s.start()


    CODEC=cv2.VideoWriter_fourcc('M','J','P','G')
    vid_capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    vid_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    vid_capture.set(cv2.CAP_PROP_FOURCC, CODEC)
    #vid_capture.set(cv2.CAP_PROP_FPS, 60)
    vid_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if mp:
        with mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence = 0.5) as face_detection:
            while vid_capture.isOpened():
                # Obtain image
                ret, image = vid_capture.read()
                boxes = []

                if not ret: continue
                h, w = image.shape[:2]

                image.flags.writeable = False

                # Pose detection with MediaPipe
                results = face_detection.process(image)
                
                # Recolor image to BGR
                image.flags.writeable = True

                # Draw detected face
                if results.detections:
                    for detection in results.detections:
                        #print(detection.location_data.relative_bounding_box)
                        det = detection.location_data.relative_bounding_box
                        #mp_drawing.draw_detection(image, detection)
                        boxes.append([int(det.xmin * w), int(det.ymin * h), int((det.width+det.xmin) * w), int((det.ymin+det.height) * h)])

                
                if boxes:
                    for box in boxes:
                        face = image[box[1]:box[3], box[0]:box[2]]
                        # Blur
                        if not True:
                            face = cv2.resize(face, (10,10))
                            face = cv2.resize(face, (box[3]-box[1],box[2]-box[0]))
                        elif True:
                            face = cv2.blur(face, (45, 45))

                        image[box[1]:box[3], box[0]:box[2]] = face
                        #cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), (255,255,255), thickness=2)
                
                web_set(image)
                #cv2.imshow('feed', image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
   

    vid_capture.release()
    cv2.destroyAllWindows()