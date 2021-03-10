#! usr/bin/env python
# *-- coding : utf-8 --*

import re
from nltk.tokenize import TweetTokenizer
from flask import Flask, request, jsonify
from vgg16 import predict_meme_class
import os
import image_processing

app = Flask(__name__)



@app.route('/post/', methods=['POST', "GET"])
def post_something():
    meme_path = request.args.get('meme')
    result_dict = {}
    if (not request.data):
        return "No data was sent !"
        # getting the image from client
    image_name = image_processing.get_image(request)

    # check if the file less than 1 MB
    if os.stat(image_name).st_size > 1000000:
        # reduce the size of the received image and delete the old one
        new_image_name = image_processing.get_image_less_than_1MB(image_name)
        os.remove(image_name)
    else:
        new_image_name = image_name
    predicted_label = predict_meme_class(new_image_name)
    result_dict['prediction_score'] = predicted_label
    return jsonify(result_dict)


@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)