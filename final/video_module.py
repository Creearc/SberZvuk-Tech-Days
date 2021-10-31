from multiprocessing import Process, Value, Queue


class Analyzer():
  def __init__(self, url):
    pass

  def start(self, cores):
    c = Process(target=self.process, args=())
    c.start()

  def process(self):


