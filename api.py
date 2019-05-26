from flask import Flask , request, jsonify
from flask_cors import CORS

import numpy as np 
import base64
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from utils import ValueInvert

# path = './tmp/' 
# get_checkpoint = tf.train.latest_checkpoint(path) 
# W1 = tf.train.load_variable(get_checkpoint, 'W1')
# W2 = tf.train.load_variable(get_checkpoint, 'W2')
# b1 = tf.train.load_variable(get_checkpoint, 'b1')

# 2 layer  NN

# sess = tf.Session()
# saver = tf.train.import_meta_graph('./tmp/2layers_test74-1000.meta')
# saver.restore(sess, tf.train.latest_checkpoint('./tmp/'))
# graph = tf.get_default_graph()
# X = graph.get_tensor_by_name("X:0")
# Y_hat = graph.get_tensor_by_name("Y_hat:0")

app = Flask(__name__)
CORS(app)

@app.route('/recognize', methods = ['POST'])
def recognize():
  request_data = request.get_json()
  imgbase64 = request_data['data']
  encoded_data = imgbase64.split(',')[1]
  filename = 'canvas_image.png'
  imgdata = base64.b64decode(encoded_data)
  with open(filename, 'wb') as f:
    f.write(imgdata)
  
  img = cv2.imread('canvas_image.png', cv2.IMREAD_GRAYSCALE)
  small = cv2.resize(img, (28, 28))
  cv2.imwrite('resize.png', small)
  data = cv2.imread('resize.png', cv2.IMREAD_GRAYSCALE)
  data = ValueInvert(data)/255.
  # Center image here
  X_1 = center_image(data)
  # print(data)
  X_1 = data.reshape(28 * 28, 1)
  # print(X_1)
  pred = sess.run([Y_hat], feed_dict={ X: X_1 })
  # pred = np.argmax(pred, axis=0)
  print(pred)
  # print(str(pred_dev))
  pred = np.argmax(pred, axis=0)
  return jsonify({'number': int(pred[0,0])}), 200

@app.route('/save_dev', methods = ['POST'])
def save_dev():
  request_data = request.get_json()
  imgbase64 = request_data['data']
  number = request_data['number']
  print(number)
  encoded_data = imgbase64.split(',')[1]
  filename = 'canvas_image.png'
  imgdata = base64.b64decode(encoded_data)
  with open(filename, 'wb') as f:
    f.write(imgdata)
  
  img = cv2.imread('canvas_image.png', cv2.IMREAD_GRAYSCALE)
  small = cv2.resize(img, (28, 28))
  cv2.imwrite('./dev_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', small)

  return jsonify({}), 200

if __name__ == '__main__':
  src = 'dev_images'
  dest = 'dev_images_centered_2'

  for number in range(9, 10):
    for filename in os.listdir(src + '/' +str(number)):
      img = cv2.imread(os.path.join(src, str(number), filename), cv2.IMREAD_GRAYSCALE)
      if img is not None:
        print('Preprocess '+filename)
        newimg = process_image(img)
        cv2.imwrite(os.path.join(dest, str(number), filename), newimg)

  # img = cv2.imread(os.path.join(src, '7', '28565061858175218.png'), cv2.IMREAD_GRAYSCALE)
  # plt.imshow(img)
  # plt.show()
  # test = process_image(img)
  # plt.imshow(test)
  # plt.show()


  # app.run(port='5002')

  app.run(port='5002')