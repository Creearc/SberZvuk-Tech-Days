import threading
from flask import Flask, request, redirect, url_for, jsonify
import json

import video_module
import buck_module


def process(url, prefix):
  video_analyzer.start(url, prefix)


  for file_name in ["{}_audio.json".format(prefix),
                    "{}_video.json".format(prefix),
                    "{}_result.mp4".format(prefix)]:

    buck_module.upl(filename=file_name,
                    s3_path=file_name)
  


app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def question_page():
  if request.method == 'POST':
    data = json.loads(request.data)
    if ('source' and 'prefix') in data.keys():
      url = data['source']
      prefix = data['prefix']
      threading.Thread(target=process, args=(url, prefix, )).start()
      text = {
        "code" : "200",
        "message" : "OK"
      }
      
    else:    
      text = {
        "code" : "400",
        "message" : "Bad Request"
        }
      
    return jsonify(text)

if __name__ == "__main__":
  video_analyzer = video_module.Analyzer()
  app.run(host='0.0.0.0', port=80, debug=True)
