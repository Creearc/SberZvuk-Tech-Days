import requests

def ask(url, data):
  response = requests.post(url='http://{}'.format(url), json=data)
  out = response.json()
  return out


if __name__ == "__main__":
  url = '46.243.142.161/recognize'
  
  data = {
    "source" : "http://hackaton.sber-zvuk.com/hackathon_part_1.mp4", 
    "prefix" : "Rostislav" 
    }
  
  print(ask(url, data))
