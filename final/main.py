import os
import threading
from flask import Flask, request, redirect, url_for, jsonify
import json

import video_module
import audio_module
import compose_video_module
import buck_module

OUTPUT_FOLDER = 'output'


def process(url, prefix):
    video_analyzer.start(url, prefix)
    audio_module.start(url, prefix)

    audio_name = "{}/tmp_{}_audio.wav".format(OUTPUT_FOLDER, prefix)
    video_name = "{}/tmp_{}_video.mp4".format(OUTPUT_FOLDER, prefix)
    compose_video_module.composit(audio_name, video_name,
                                  "{}_result.mp4".format(prefix))

    for file_name in ["{}/{}_audio.json".format(OUTPUT_FOLDER, prefix),
                      #"{}/{}_video.json".format(prefixOUTPUT_FOLDER, ),
                      "{}/{}_result.mp4".format(OUTPUT_FOLDER, prefix)]:
        buck_module.upl(filename=file_name, s3_path=file_name.split('/'[0]))

    files_to_remove = ["{}/{}_audio.json".format(OUTPUT_FOLDER, prefix),
                       #"{}/{}_video.json".format(prefixOUTPUT_FOLDER, ),
                       "{}/{}_result.mp4".format(OUTPUT_FOLDER, prefix),
                       audio_name,
                       video_name]

    print(os.listdir(OUTPUT_FOLDER))

    for file_name in files_to_remove:
        os.remove(file_name)


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
                "code": "200",
                "message": "OK"
            }

        else:
            text = {
                "code": "400",
                "message": "Bad Request"
                }

        return jsonify(text)

if __name__ == "__main__":
    video_analyzer = video_module.Analyzer()
    app.run(host='0.0.0.0', port=80, debug=True)
