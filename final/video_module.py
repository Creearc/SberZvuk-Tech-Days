from multiprocessing import Process, Value, Queue


class Analyzer():
  def __init__(self, url):
    self.vid_capture = cv2.VideoCapture(url)
    self.frame_count = int(self.vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))

  def start(self, cores):
    c = Process(target=self.process, args=())
    c.start()

  def process(self):


