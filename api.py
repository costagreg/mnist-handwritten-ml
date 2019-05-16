from flask import Flask , request, jsonify
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf


# path = './tmp/' 
# get_checkpoint = tf.train.latest_checkpoint(path) 
# W1 = tf.train.load_variable(get_checkpoint, 'W1')
# W2 = tf.train.load_variable(get_checkpoint, 'W2')
# b1 = tf.train.load_variable(get_checkpoint, 'b1')

layer_1 = 784
layer_2 = 20
layer_3 = 10

W1 = tf.Variable(tf.random.normal([layer_2, layer_1]), dtype=tf.float32, name='W1')
W2 = tf.Variable(tf.random.normal([layer_3, layer_2]), dtype=tf.float32, name='W2')
b1 = tf.Variable(np.zeros((layer_2, 1)), dtype=tf.float32, name='b1')
b2 = tf.Variable(np.zeros((layer_3, 1)), dtype=tf.float32, name='b2')

X = tf.placeholder(tf.float32, shape=[layer_1, None ], name= 'X')
Y = tf.placeholder(tf.float32, shape=[layer_3, None ], name= 'Y')

Z1 = tf.add(tf.matmul(W1, X), b1) # [layer_2, None]
A1 = tf.nn.sigmoid(Z1)
Z2 = tf.add(tf.matmul(W2, A1), b2) # [layer_3, None]
Y_hat = tf.nn.softmax(Z2)

imported_graph = tf.train.import_meta_graph('./tmp/mini_batch-23000.meta')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
imported_graph.restore(sess, './tmp/mini_batch-23000')

app = Flask(__name__)
@app.route('/recognize', methods = ['POST'])
def recognize():
  request_data = request.get_json()
  img = np.array(request_data['data'])
  X_1 = img.reshape(img.shape[0], 1)
  X_1 = X_1/255.
  print(X_1)
  test, pred_test = sess.run([Z2, Y_hat], feed_dict={ X: X_1 })
  print(test)
  print(np.argmax(test, axis=0))
  return jsonify({'number':1}), 200

if __name__ == '__main__':
  app.run(port='5002')
