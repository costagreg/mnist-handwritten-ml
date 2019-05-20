from flask import Flask , request, jsonify
from flask_cors import CORS

import numpy as np 
import base64
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
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
  # Add padding of 100 px and translate image
  img_with_padding = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255,255,255])
  img_with_padding_resized = cv2.resize(img_with_padding, (28,28))

  rows, cols = img_with_padding.shape
  
  # Translate images
  M1 = np.float32([[1,0,30],[0,1,0]])
  M2 = np.float32([[1,0,0],[0,1,30]])
  M3 = np.float32([[1,0,-30],[0,1,0]])
  M4 = np.float32([[1,0,0],[0,1,-30]])

  M1_img = cv2.warpAffine(img_with_padding,M1,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])
  M2_img = cv2.warpAffine(img_with_padding,M2,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])
  M3_img = cv2.warpAffine(img_with_padding,M3,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])
  M4_img = cv2.warpAffine(img_with_padding,M4,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])

  M1_img_resized = cv2.resize(M1_img, (28,28))
  M2_img_resized = cv2.resize(M2_img, (28,28))
  M3_img_resized = cv2.resize(M3_img, (28,28))
  M4_img_resized = cv2.resize(M4_img, (28,28))
  
  # Rotate image matrix 
  R_1 = cv2.getRotationMatrix2D((cols/2,rows/2),15,1)
  R_2 = cv2.getRotationMatrix2D((cols/2,rows/2),-15,1)


  M1_R_1_img = cv2.warpAffine(M1_img,R_1,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])
  M1_R_2_img = cv2.warpAffine(M1_img,R_2,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])

  M2_R_1_img = cv2.warpAffine(M2_img,R_1,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])
  M2_R_2_img = cv2.warpAffine(M2_img,R_2,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])

  M3_R_1_img = cv2.warpAffine(M3_img,R_1,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])
  M3_R_2_img = cv2.warpAffine(M3_img,R_2,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])

  M4_R_1_img = cv2.warpAffine(M4_img,R_1,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])
  M4_R_2_img = cv2.warpAffine(M4_img,R_2,(cols,rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[255,255,255])

  M1_R_1_img_resized = cv2.resize(M1_R_1_img, (28,28))
  M1_R_2_img_resized  = cv2.resize(M1_R_2_img, (28,28))
  M2_R_1_img_resized  = cv2.resize(M2_R_1_img, (28,28))
  M2_R_2_img_resized  = cv2.resize(M2_R_2_img, (28,28))
  M3_R_1_img_resized = cv2.resize(M3_R_1_img, (28,28))
  M3_R_2_img_resized = cv2.resize(M3_R_2_img, (28,28))
  M4_R_1_img_resized = cv2.resize(M4_R_1_img, (28,28))
  M4_R_2_img_resized = cv2.resize(M4_R_2_img, (28,28))

  cv2.imwrite('img_with_padding_resized.png', img_with_padding_resized)
  cv2.imwrite('M1_img.png', M1_img_resized)
  cv2.imwrite('M2_img.png', M2_img_resized)
  cv2.imwrite('M3_img.png', M3_img_resized)
  cv2.imwrite('M4_img.png', M4_img_resized)
  cv2.imwrite('M1_R_1_img.png', M1_R_1_img_resized)
  cv2.imwrite('M1_R_2_img.png', M1_R_2_img_resized)
  cv2.imwrite('M2_R_1_img.png', M2_R_1_img_resized)
  cv2.imwrite('M2_R_2_img.png', M2_R_2_img_resized)
  cv2.imwrite('M3_R_1_img.png', M3_R_1_img_resized)
  cv2.imwrite('M3_R_2_img.png', M3_R_2_img_resized)
  cv2.imwrite('M4_R_1_img.png', M4_R_1_img_resized)
  cv2.imwrite('M4_R_2_img.png', M4_R_2_img_resized)

  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', small)
  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', img_with_padding_resized)
  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', M1_img_resized)
  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', M2_img_resized)
  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', M3_img_resized)
  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', M4_img_resized)
  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', M1_R_1_img_resized)
  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', M1_R_2_img_resized)
  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', M2_R_1_img_resized)
  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', M2_R_2_img_resized)
  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', M3_R_1_img_resized)
  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', M3_R_2_img_resized)
  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', M4_R_1_img_resized)
  cv2.imwrite('./training_images/'+str(number)+'/'+ str(np.random.randint(0, high=100000000000000000)) +'.png', M4_R_2_img_resized)
  
  return jsonify({}), 200

if __name__ == '__main__':
  app.run(port='5002')
