#! usr/bin/env python
# *-- coding : utf-8 --*

import re
from nltk.tokenize import TweetTokenizer
from flask import Flask, request, jsonify
from vgg16 import predict_meme_class

app = Flask(__name__)



@app.route('/post/', methods=['POST', "GET"])
def post_something():
    meme_path = request.args.get('meme')
    result_dict = {}
    predicted_label = predict_meme_class(meme_path)
    result_dict['prediction_score'] = predicted_label
    return jsonify(result_dict)


@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)