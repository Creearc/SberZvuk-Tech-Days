# -*- coding: cp1251 -*-
from ftplib import FTP
import sys

PATH = '\\'.join(sys.argv[0].split('\\')[:-1])

ftp = FTP()
HOST = '192.168.68.128'
#HOST = '192.168.137.164'
PORT = 21

ftp.connect(HOST, PORT)

print(ftp.login(user='alexandr', passwd='9'))


ftp.cwd('~/SberZvuk-Tech-Days/final/models')
#ftp.cwd('~/SberZvuk-Tech-Days/opencv-face-recognition')

#for i in range(28, 36):
if True:
  fl = 'embeddings.pickle'
  fl = 'le.pickle'
  #fl = 'recognizer.pickle'
  #fl = 'openface_nn4.small2.v1.t7'
  out = '{}\{}'.format(PATH, fl)

  with open(out, 'wb') as f:
      ftp.retrbinary('RETR ' + fl, f.write)

