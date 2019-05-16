import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from mnist import get_test, get_train 
from utils import hot_encoding, classification_rate, prepare_Y, prepare_X

np.set_printoptions(threshold=np.inf)

num_train = 60000
num_test = 10000

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

# 2 layer  NN

layer_1 = 784
layer_2 = 20
layer_3 = 10

# Define weights and bias

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

loss = tf.losses.log_loss(labels=Y, predictions=Y_hat)
opt = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

saver = tf.train.Saver()
# tf.reset_default_graph()
imported_graph = tf.train.import_meta_graph('./tmp/batch_gradient-29000.meta')

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
  sess.run(tf.global_variables_initializer())
  imported_graph.restore(sess, './tmp/batch_gradient-29000')
  for i in range(29001, 40000):
    pred, cost, _ = sess.run([Y_hat, loss, opt], feed_dict={ X: X_train, Y: Y_train_E})
    if i%1000 == 0:
      saver.save(sess, './tmp/batch_gradient', global_step=i)
      print('iter '+str(i))
      print('cost '+str(cost))
      pred_test, cost, _ = sess.run([Y_hat, loss, opt], feed_dict={ X: X_test, Y: Y_test_E })
      pred_test = np.argmax(pred_test, axis=0)
      print('class' + str(classification_rate(Y_test, pred_test)))
