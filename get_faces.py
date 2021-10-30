import os
import cv2

PATH = 'C:/Users/Саша/Downloads/vox1_dev_txt/txt'

URL = "https://www.youtube.com/watch?v={}"

for idd in os.listdir(PATH):
  for url in os.listdir('{}/{}'.format(PATH, idd)):
    skip = False
    vid_capture = cv2.VideoCapture(URL.format(url))
    
    for file in os.listdir('{}/{}/{}'.format(PATH, idd, url)):
      if skip:
        f.close()
        break
      
      f = open('{}/{}/{}/{}'.format(PATH, idd, url, file))
      for l in range(8):
        f.readline()
        
      for l in f:
        index, x, y, w, h = [int(i) for i in l.split()]
        index = int(index)
        print(index, x, y, w, h)
        
        vid_capture.set(1, index)
        ret, image = vid_capture.read()
        if image is None:
          skip = True
          break

        cv2.imshow('feed', image[y : y + h, x : x + w])
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
      f.close()
    
