from flask import Flask , request, jsonify
from flask_cors import CORS

import numpy as np 
import base64
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from utils import ValueInvert


# path = './tmp/' 
# get_checkpoint = tf.train.latest_checkpoint(path) 
# W1 = tf.train.load_variable(get_checkpoint, 'W1')
# W2 = tf.train.load_variable(get_checkpoint, 'W2')
# b1 = tf.train.load_variable(get_checkpoint, 'b1')

def data_uri_to_cv2_img(uri):
  encoded_data = uri.split(',')[1]
  filename = 'canvas_image.png'
  imgdata = base64.b64decode(encoded_data)
  with open(filename, 'wb') as f:
    f.write(imgdata)
  
  img = cv2.imread('canvas_image.png', 0)

  return img


# 2 layer  NN

# sess = tf.Session()
# saver = tf.train.import_meta_graph('./tmp/2layers_test74-1000.meta')
# saver.restore(sess, tf.train.latest_checkpoint('./tmp/'))
# graph = tf.get_default_graph()
# X = graph.get_tensor_by_name("X:0")
# Y_hat = graph.get_tensor_by_name("Y_hat:0")

def center_dev_images():
  src = 'dev_images'
  dest = 'dev_images_centered'

  for number in range(0, 9):
    for filename in os.listdir(src + '/' +str(number)):
      img = cv2.imread(os.path.join(src, str(number), filename), cv2.IMREAD_GRAYSCALE)
      if img is not None:
        # img = ValueInvert(img)
        # X.append(img.reshape(image_size * image_size))
        # Y.append(number)
        cv2.imwrite(os.path.join(dest, str(number), filename), center_image(img))

def find_center_image(img):
  left = 0
  right = img.shape[1] - 1

  empty_left = True
  empty_right = True

  for col in range(int(img.shape[1])):
    if empty_left == False and empty_right == False:
      break

    for row in range(img.shape[0] - 1):
      if img[row, col] < 255 and empty_left == True:
        empty_left = False
        left = col

      if img[row, img.shape[1] - col - 1] < 255 and empty_right == True:
        empty_right = False
        right = img.shape[1] - col


  top = 0
  bottom = img.shape[0] - 1

  empty_top = True
  empty_bottom = True

  for row in range(int(img.shape[0])):
    if empty_top == False and empty_bottom == False:
      break

    for col in range(img.shape[1] - 1):
      if img[row, col] < 255 and empty_top == True:
        empty_top = False
        top = row

      if img[img.shape[0] - row - 1, col] < 255 and empty_bottom == True:
        empty_bottom = False
        bottom = img.shape[0] - row

  return top, right, bottom, left

def center_image(img):
  cols, rows = img.shape
  top, right, bottom, left = find_center_image(img)
  width = right - left
  height = bottom - top
  center_digit_x = (width / 2) + left
  center_digit_y = (height / 2) + top
  center_img_x = cols / 2
  center_img_y = rows / 2
  t_x = center_img_x - center_digit_x
  t_y = center_img_y - center_digit_y

  # print(t_x)
  # print(t_y)
  M = np.float32([[1,0,t_x],[0,1,t_y]])
  dst = cv2.warpAffine(img,M,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])
  # plt.imshow(dst)
  # plt.show()
  return dst

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
  print(X_1)
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


  app.run(port='5002')