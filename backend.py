from flask import Flask, request, redirect, url_for, jsonify
import json

app = Flask(__name__)

@app.route('/recognize', methods=['POST'])
def question_page():
  if request.method == 'POST':
    text = {
      "code" : "200",
      "message" : "OK"
      }
    return jsonify(text)

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=80, debug=True)
