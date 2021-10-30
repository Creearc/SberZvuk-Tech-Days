import cv2
import numpy as np
import time
import requests

ip = "http://hackaton.sber-zvuk.com/hackathon_part_1.mp4"
#resolution=(1920, 1080)
resolution=(1280, 720)

CODEC=cv2.VideoWriter_fourcc('M','J','P','G')
vid_capture = cv2.VideoCapture(ip)
vid_capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
vid_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
vid_capture.set(cv2.CAP_PROP_FOURCC, CODEC)
vid_capture.set(cv2.CAP_PROP_FPS, 60)
vid_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while vid_capture.isOpened():
    # Obtain image
    ret, image = vid_capture.read()
    if not ret: continue
    h, w = image.shape[:2]
    
    image = cv2.resize(image, (1280,720))

    

    cv2.imshow('feed', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid_capture.release()
cv2.destroyAllWindows()