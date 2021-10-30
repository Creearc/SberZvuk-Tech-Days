import os
import cv2
import pafy
import threading

PATH = 'txt'
OUTPUT = 'dataset'

URL = "https://www.youtube.com/watch?v={}"

def ppp(idd):
  if not os.path.isdir('{}/{}'.format(OUTPUT, idd)):
    os.mkdir('{}/{}'.format(OUTPUT, idd))
  for url in os.listdir('{}/{}'.format(PATH, idd)):
    skip = False
    print(URL.format(url))
    try:
      videoPafy = pafy.new(URL.format(url))
      play = videoPafy.getbest(preftype="mp4")
      vid_capture = cv2.VideoCapture(play.url)
    except:
      continue
    frame_count = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    
    for file in os.listdir('{}/{}/{}'.format(PATH, idd, url)):
      if skip:
        f.close()
        vid_capture.release()
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

        name = '{}/{}'.format(OUTPUT, idd)
        cv2.imwrite('{}/{}/{}.jpg'.format(OUTPUT, idd, len(os.listdir('{}/{}'.format(OUTPUT, idd)))),
                    image[y : y + h, x : x + w])
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
      f.close()
      vid_capture.release()
      

for idd in os.listdir(PATH):
  threading.Thread(target=ppp, args=(idd, )).start()
  
  
    

