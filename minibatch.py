import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

from mnist import get_test, get_train, get_dev
from utils import hot_encoding, classification_rate, prepare_Y, prepare_X, read_variable_from_batch

# np.set_printoptions(threshold=np.inf)

test_name, hidden_layer, learning_rate = read_variable_from_batch()

print('---------------')
print('Test name:' + test_name)
print('Hidden_layer:' + str(hidden_layer))
print('Learning_rate:' + str(learning_rate))
print('---------------')

num_train = 60000
num_test = 10000
batch_size = 32

# Training data
X_train, Y_train = get_train(num_train)
X_train = prepare_X(X_train)
Y_train, Y_train_E = prepare_Y(Y_train, 10)

print('X_train shape ' + str(X_train.shape))
print('Y_train shape ' + str(Y_train_E.shape))

# Test data
X_test, Y_test = get_test(num_test)
X_test = prepare_X(X_test)
Y_test, Y_test_E = prepare_Y(Y_test, 10)

print('X_test shape ' + str(X_test.shape))
print('Y_test shape ' + str(Y_test_E.shape))

# Dev data
X_dev, Y_dev = get_dev()
X_dev = prepare_X(X_dev)
Y_dev, Y_dev_E = prepare_Y(Y_dev, 10)

print('X_dev shape ' + str(X_dev.shape))
print('Y_dev shape ' + str(Y_dev_E.shape))

# 2 layer  NN

layer_1 = 784
layer_2 = hidden_layer
layer_3 = 10

# Define weights and bias

W1 = tf.Variable(tf.random.uniform([layer_2, layer_1], minval=0.01, maxval=0.05), dtype=tf.float32, name='W1')
W2 = tf.Variable(tf.random.uniform([layer_3, layer_2], minval=0.01, maxval=0.05), dtype=tf.float32, name='W2')
b1 = tf.Variable(np.zeros((layer_2, 1)), dtype=tf.float32, name='b1')
b2 = tf.Variable(np.zeros((layer_3, 1)), dtype=tf.float32, name='b2')

X = tf.placeholder(tf.float32, shape=[layer_1, None ], name= 'X')
Y = tf.placeholder(tf.float32, shape=[layer_3, None ], name= 'Y')

Z1 = tf.add(tf.matmul(W1, X), b1) # [layer_2, None]
A1 = tf.nn.relu(Z1)
Z2 = tf.add(tf.matmul(W2, A1), b2) # [layer_3, None]
Y_hat = tf.nn.softmax(Z2)

loss = tf.losses.log_loss(labels=Y, predictions=Y_hat)
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()
# tf.reset_default_graph()
# imported_graph = tf.train.import_meta_graph('./tmp/mini_batch-14000.meta')

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # imported_graph.restore(sess, './tmp/mini_batch-26000')
  # X_train = create_batches(X_train)
  # (num_batches, layer_1, m)
  num_batches = int(np.ceil(X_train.shape[1]/batch_size))
  for i in range(30000):
    for j in range(num_batches):
      batch_start = batch_size*j
      batch_end = batch_size*(j+1)
      X_batch =  X_train[:, batch_start:batch_end]
      Y_batch =  Y_train_E[:, batch_start:batch_end]
      pred, cost, _ = sess.run([Y_hat, loss, opt], feed_dict={ X: X_batch, Y: Y_batch })

    if i%100 == 0:
      saver.save(sess, './tmp/' + test_name, global_step=i)
      print('iter '+str(i))
      print('cost '+str(cost))
      pred_dev, cost, _ = sess.run([Y_hat, loss, opt], feed_dict={ X: X_dev, Y: Y_dev_E })
      pred_dev = np.argmax(pred_dev, axis=0)
      print('dev class ' + str(classification_rate(Y_dev, pred_dev)))
  
  pred_test, cost, _ = sess.run([Y_hat, loss, opt], feed_dict={ X: X_test, Y: Y_test_E })
  pred_test = np.argmax(pred_test[0][0], axis=0)
  print('class' + str(classification_rate(Y_test, pred_test)))