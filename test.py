import cv2
import numpy as np
import time
import os


def HAAR(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = cascade.detectMultiScale(image, scaleFactor=1.1,
                                       minNeighbors=2, minSize=(30, 30),
                                       flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0: return None
    rects[:,2:] += rects[:,:2]
    #box = None
    boxes = []
    maximum = None
    for x1, y1, x2, y2 in rects:
      size = ((x2-x1) * (y2-y1))
      if maximum is None or maximum < size:
        maximum = size
        boxes.append([x1, y1, x2, y2])
        #box = [x1, y1, x2, y2]
    return boxes



ip = "https://www.youtube.com/watch?v={}"
#ip = "http://hackaton.sber-zvuk.com/hackathon_part_1.mp4"
haar_path = "face.xml"

vid_capture = cv2.VideoCapture(ip)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(1, index)


'''
while vid_capture.isOpened():
    # Obtain image
    ret, image = vid_capture.read()
    if not ret: continue
    h, w = image.shape[:2]

    # Resize
    image = cv2.resize(image, (720,405))

    cascade = cv2.CascadeClassifier(haar_path)    
    boxes = HAAR(image)
    
    if boxes:
        for box in boxes:
            face = image[box[1]:box[3], box[0]:box[2]]
            # Blur
            if True:
                face = cv2.resize(face, (10,10))
                face = cv2.resize(face, (box[3]-box[1],box[2]-box[0]))
            elif True:
                face = cv2.blur(face, (15, 15))

            image[box[1]:box[3], box[0]:box[2]] = face
            #cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), (255,255,255), thickness=2)

    cv2.imshow('feed', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

vid_capture.release()
cv2.destroyAllWindows()