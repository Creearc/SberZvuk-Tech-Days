import os
import cv2
import numpy as np
import pickle
import multiprocessing
from multiprocessing import Process, Value, Queue

FACE_CONFIDENCE = 0.5
RECOGNITION_CONFIDENCE = 0.01
CPU_COUNT = multiprocessing.cpu_count()

class Analyzer():
  def __init__(self):
    protoPath = "models/deploy.prototxt"
    modelPath = "models/res10_300x300_ssd_iter_140000.caffemodel"

    self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    self.embedder = cv2.dnn.readNetFromTorch("models/openface_nn4.small2.v1.t7")

    self.recognizer = pickle.loads(open("models/recognizer.pickle", "rb").read())
    self.le = pickle.loads(open("models/le.pickle", "rb").read())

  def start(self, url, cores=1):   
    self.vid_capture = cv2.VideoCapture(url)
    self.frame_count = int(self.vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(cores):
      part = self.frame_count // cores
      start = part * i
      if i == cores - 1:
        end = self.frame_count
      else:
        end = part * (i + 1)
      c = Process(target=self.process, args=(start, end, ))
      c.start()
      c.join()

    self.vid_capture.release()      
    print('Complite')

  def process(self, start, end):
    for i in range(start, end):
      self.vid_capture.set(1, i)
      _, frame = self.vid_capture.read()
      if frame is None:
        continue

      h, w = frame.shape[:2]
      imageBlob = cv2.dnn.blobFromImage(
	cv2.resize(frame, (300, 300)), 1.0, (300, 300),
	(104.0, 177.0, 123.0), swapRB=False, crop=False)

      self.detector.setInput(imageBlob)
      detections = self.detector.forward()

      for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > FACE_CONFIDENCE:
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")

          face = frame[startY:endY, startX:endX]
          (fH, fW) = face.shape[:2]

          if fW < 20 or fH < 20:
            continue

          faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                           (0, 0, 0), swapRB=True, crop=False)
          self.embedder.setInput(faceBlob)
          vec = self.embedder.forward()
          preds = self.recognizer.predict_proba(vec)[0]

          j = np.argmax(preds)
          proba = preds[j]
          name = self.le.classes_[j]

          if proba > RECOGNITION_CONFIDENCE:
            text = "{}: {:.2f}%".format(name, proba * 100)
            
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            face = cv2.resize(face, (5, 5))
            face = cv2.resize(face, (fW, fH))

            frame[startY:endY, startX:endX] = face


if __name__ == '__main__':
  url = "http://hackaton.sber-zvuk.com/hackathon_part_1.mp4"
  a = Analyzer()
  a.start(url, CPU_COUNT)
