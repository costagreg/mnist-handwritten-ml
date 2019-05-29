import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

from data import  get_all_MNSIT, get_CANVAS
from utils import unison_shuffled_copies, hot_encoding, classification_rate, prepare_Y, prepare_X, read_variable_from_batch

# np.set_printoptions(threshold=np.inf)

test_name, hidden_layer, learning_rate = read_variable_from_batch()

print('---------------')
print('Test name:' + test_name)
print('Hidden_layer:' + str(hidden_layer))
print('Learning_rate:' + str(learning_rate))
print('---------------')

batch_size = 128

X_minst, Y_minst = get_all_MNSIT()
X_canvas_train, Y_canvas_train, X_dev, Y_dev, X_test, Y_test = get_CANVAS()

# Training data
X_train = np.concatenate((X_minst, X_canvas_train))
Y_train = np.concatenate((Y_minst, Y_canvas_train))

X_train, Y_train = unison_shuffled_copies(X_train, Y_train)

X_train_dev = X_train[0:X_dev.shape[0],:]
Y_train_dev = Y_train[0:X_dev.shape[0]]

X_train = X_train[X_dev.shape[0]:,:]
Y_train = Y_train[X_dev.shape[0]:]

X_train = prepare_X(X_train)
Y_train, Y_train_E = prepare_Y(Y_train, 10)

print('X_train shape ' + str(X_train.shape))
print('Y_train shape ' + str(Y_train_E.shape))

X_train_dev = prepare_X(X_train_dev)
Y_train_dev, Y_train_dev_E = prepare_Y(Y_train_dev, 10)

print('X_train_dev shape ' + str(X_train_dev.shape))
print('Y_train_dev shape ' + str(Y_train_dev_E.shape))

# Test data

X_test = prepare_X(X_test)
Y_test, Y_test_E = prepare_Y(Y_test, 10)

print('X_test shape ' + str(X_test.shape))
print('Y_test shape ' + str(Y_test_E.shape))

# Dev data
X_dev = prepare_X(X_dev)
Y_dev, Y_dev_E = prepare_Y(Y_dev, 10)

print('X_dev shape ' + str(X_dev.shape))
print('Y_dev shape ' + str(Y_dev_E.shape))

# 2 layer  NN

layer_1 = 784
layer_2 = hidden_layer
layer_3 = hidden_layer
layer_4 = 10

# Define weights and bias

W1 = tf.Variable(tf.random.normal([layer_1, layer_2]) * np.sqrt(2/layer_1), dtype=tf.float32, name='W1')
W2 = tf.Variable(tf.random.normal([layer_2, layer_3]) * np.sqrt(2/layer_2), dtype=tf.float32, name='W2')
W3 = tf.Variable(tf.random.normal([layer_3, layer_4]) * np.sqrt(2/layer_3), dtype=tf.float32, name='W3')
b1 = tf.Variable(np.zeros((1, layer_2)), dtype=tf.float32, name='b1')
b2 = tf.Variable(np.zeros((1, layer_3)), dtype=tf.float32, name='b2')
b3 = tf.Variable(np.zeros((1, layer_4)), dtype=tf.float32, name='b3')

X = tf.placeholder(tf.float32, shape=[None, layer_1], name= 'X')
Y = tf.placeholder(tf.float32, shape=[None, layer_4], name= 'Y')

Z1 = tf.add(tf.matmul(X, W1), b1, name='Z1') # [None, layer_2]
A1 = tf.nn.relu(Z1, name='A1')
Z2 = tf.add(tf.matmul(A1, W2), b2, name='Z2') #[None, layer_3]
A2 = tf.nn.relu(Z2, name='A2')
Z3 = tf.add(tf.matmul(A2, W3), b3, name='Z3') #[None, layer_4]

# L2 regularization
regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Z3))
loss = loss + 0.01 * regularizers

opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()
# tf.reset_default_graph()
imported_graph = tf.train.import_meta_graph('./training_1/2layers_test800_00001-8800.meta')

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  imported_graph.restore(sess, tf.train.latest_checkpoint('./training_1'))
  graph = tf.get_default_graph()
  num_batches = int(np.ceil(X_train.shape[0]/batch_size))
  for i in range(30000):
    for j in range(num_batches):
      batch_start = batch_size*j
      batch_end = batch_size*(j+1)
      X_batch =  X_train[batch_start:batch_end,:]
      Y_batch =  Y_train_E[batch_start:batch_end,:]
      pred, cost, _ = sess.run([Z3, loss, opt], feed_dict={ X: X_batch, Y: Y_batch })

    if i%200 == 0:
      saver.save(sess, './tmp/2layers_' + test_name, global_step=i)
      pred_dev = sess.run(Z3, feed_dict={ X: X_dev, Y: Y_dev_E })
      pred_dev = np.argmax(pred_dev, axis=1)
      misclassified, class_rate = classification_rate(Y_dev, pred_dev)
      print('--------------------')
      print('---| iter '+str(i))
      print('---| cost '+str(cost))
      print('---| dev class '+str(class_rate))
      print('---| dev misclassified '+str(misclassified))

    if i%1000 == 0:
      pred_train_dev, cost, _ = sess.run([Z3, loss, opt], feed_dict={ X: X_train_dev, Y: Y_train_dev_E })
      pred_train_dev = np.argmax(pred_train_dev, axis=1)
      misclassified, class_rate = classification_rate(Y_train_dev, pred_train_dev)
      print('---| train-dev class ' + str(class_rate))
      print('---| train-dev misclassified ' + str(misclassified))
  
  pred_test, cost, _ = sess.run([Z3, loss, opt], feed_dict={ X: X_test, Y: Y_test_E })
  pred_test = np.argmax(pred_test, axis=1)
  print('class' + str(classification_rate(Y_test, pred_test)))