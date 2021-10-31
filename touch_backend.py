import requests

def ask(url, data):
  response = requests.post(url='http://{}'.format(url), json=data)
  out = response.json()
  return out


if __name__ == "__main__":
  url = '46.243.142.161/recognize'
  
  data = {
    "source" : "https://images.all-free-download.com/footage_preview/mp4/stand_up_paddling_scuba_diving_836.mp4", 
    "prefix" : "Rostislav" 
    }
  
  print(ask(url, data))
