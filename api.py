from flask import Flask , request, jsonify
from flask_cors import CORS
from preprocess import process_image

import numpy as np 
import base64
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2

from utils import ValueInvert

path = './training_1/' 
get_checkpoint = tf.train.latest_checkpoint(path) 
W1 = tf.train.load_variable(get_checkpoint, 'W1')
W2 = tf.train.load_variable(get_checkpoint, 'W2')
W3 = tf.train.load_variable(get_checkpoint, 'W3')
b1 = tf.train.load_variable(get_checkpoint, 'b1')
b2 = tf.train.load_variable(get_checkpoint, 'b2')
b3 = tf.train.load_variable(get_checkpoint, 'b3')


sess = tf.Session()
saver = tf.train.import_meta_graph('./training_1/2layers_test800_00001-8400.meta')
saver.restore(sess, tf.train.latest_checkpoint('./training_1'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
Y_hat = graph.get_tensor_by_name("Z3:0")

app = Flask(__name__)
CORS(app)

def export_mismatch():
  src = 'canvas_images'
  dest = 'mismatch'
  mismatch = 0
  match = 0
  for number in range(0, 10):
    for filename in os.listdir(src + '/' +str(number)):
      img = cv2.imread(os.path.join(src, str(number), filename), cv2.IMREAD_GRAYSCALE)
      if img is not None:
        # print('Preprocess '+filename)
        newimg = process_image(img)
        newimg = newimg.reshape(1, 28*28)/255.
        pred = sess.run(Y_hat, feed_dict={ X: newimg })
        pred = np.argmax(pred, axis=1)
        number_pred = pred[0]

        if number_pred == number:
          match += 1
        else:
          mismatch +=1
          cv2.imwrite(os.path.join(dest, str(number), filename), img)
  
  print('Match '+ str(match))
  print('Mismatch '+ str(mismatch))
  print('Error rate '+ str(mismatch/(match+mismatch)))
  
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
  img = process_image(img)
  img = img.reshape(1, 28 * 28)/255.
  print(img)
  # print(str(pred_dev))
  pred = sess.run(Y_hat, feed_dict={ X: img })
  print(pred)
  pred = np.argmax(pred, axis=1)
  print(pred)
  return jsonify({'number': int(pred[0])}), 200

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


@app.route('/save_training', methods = ['POST'])
def save_training():
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
  img = cv2.resize(img, (28, 28))

  rows, cols = img.shape
  # Rotate image matrix 
  R_1 = cv2.getRotationMatrix2D((cols/2,rows/2),15,1)
  R_2 = cv2.getRotationMatrix2D((cols/2,rows/2),-15,1)
  R_3 = cv2.getRotationMatrix2D((cols/2,rows/2),25,1)
  R_4 = cv2.getRotationMatrix2D((cols/2,rows/2),-25,1)

  rot_1 = cv2.warpAffine(img,R_1,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])
  rot_2 = cv2.warpAffine(img,R_2,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])
  rot_3 = cv2.warpAffine(img,R_3,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])
  rot_4 = cv2.warpAffine(img,R_4,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])
 

  cv2.imwrite('./canvas_image_training/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', ValueInvert(process_image(rot_1)))
  cv2.imwrite('./canvas_image_training/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', ValueInvert(process_image(rot_2)))
  cv2.imwrite('./canvas_image_training/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', ValueInvert(process_image(rot_3)))
  cv2.imwrite('./canvas_image_training/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', ValueInvert(process_image(rot_4)))
  
  return jsonify({}), 200

if __name__ == '__main__':
  # src = 'dev_images'
  # dest = 'dev_images_centered_2'

  # for number in range(9, 10):
  #   for filename in os.listdir(src + '/' +str(number)):
  #     img = cv2.imread(os.path.join(src, str(number), filename), cv2.IMREAD_GRAYSCALE)
  #     if img is not None:
  #       print('Preprocess '+filename)
  #       newimg = process_image(img)
  #       cv2.imwrite(os.path.join(dest, str(number), filename), newimg)

  # img = cv2.imread(os.path.join(src, '7', '28565061858175218.png'), cv2.IMREAD_GRAYSCALE)
  # plt.imshow(img)
  # plt.show()
  # test = process_image(img)
  # plt.imshow(test)
  # plt.show()
  # export_mismatch()

  app.run(port='5002')