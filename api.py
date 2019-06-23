from flask import Flask, request, jsonify
from flask_cors import CORS
from preprocess import process_image

import numpy as np
import base64
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2

from utils import ValueInvert

sess = tf.Session()
saver = tf.train.import_meta_graph('./trained_model/training-2800.meta')
saver.restore(sess, tf.train.latest_checkpoint('./trained_model'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
Y_hat = graph.get_tensor_by_name("MatMul_2:0")

app = Flask(__name__)
CORS(app)


@app.route('/raecognize', methods=['POST'])
def recognize():
    request_data = request.get_json()
    imgbase64 = request_data['data']
    encoded_data = imgbase64.split(',')[1]
    filename = 'canvas_image.png'
    imgdata = base64.b64decode(encoded_data)
    with open(filename, 'wb') as f:
        f.write(imgdata)

    img = cv2.imread('canvas_image.png', cv2.IMREAD_GRAYSCALE)
    img = process_image(img)
    img = img.reshape(1, 28, 28, 1)/255.
    img = np.pad(img, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    pred = sess.run(Y_hat, feed_dict={X: img})
    pred_softmax = pred/pred.sum(axis=1, keepdims=True)
    pred = np.argmax(pred_softmax, axis=1)

    return jsonify({'number': int(pred[0])}), 200


if __name__ == '__main__':

    app.run(port='5002')
