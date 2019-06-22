import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

from data import  get_data
from utils import classification_rate,  read_variable_from_console
from models import le_net_5
# np.set_printoptions(threshold=np.inf)
# Read from comand line parameters: e.g. python3 model.py model_name hiddenLayer learningRate
test_name, hidden_layer, learning_rate = read_variable_from_console()

print('---------------')
print('Test name:' + test_name)
print('Hidden_layer:' + str(hidden_layer))
print('Learning_rate:' + str(learning_rate))
print('---------------')

batch_size = 128

# Destracture data from MINST and CANVAS
data = get_data(print_shape=True)
# Training data
X_train = data['X_train']
Y_train = data['X_train']
Y_train_E = data['Y_train_E']
# Training-dev
X_train_dev = data['X_train_dev']
Y_train_dev = data['Y_train_dev']
Y_train_dev_E = data['Y_train_dev_E']
# Dev data
X_dev = data['X_dev']
Y_dev = data['Y_dev']
Y_dev_E = data['Y_dev_E']
# Dev data
X_test = data['X_test']
Y_test = data['Y_test']
Y_test_E = data['Y_test_E']


# 2 layer  NN
X = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name= 'X')
Y = tf.placeholder(tf.float32, shape=[None, 10], name= 'Y')

Y_hat = le_net_5(X, 10)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=Y_hat))
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
saver = tf.train.Saver()
# tf.reset_default_graph()
# imported_graph = tf.train.import_meta_graph('./cn_test_1/training-2800.meta')

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # imported_graph.restore(sess, tf.train.latest_checkpoint('./cn_test_1'))
  # graph = tf.get_default_graph()
  num_batches = int(np.ceil(X_train.shape[0]/batch_size))
  for i in range(30000):
    for j in range(num_batches):
      batch_start = batch_size*j
      batch_end = batch_size*(j+1)
      X_batch =  X_train[batch_start:batch_end,:]
      Y_batch =  Y_train_E[batch_start:batch_end,:]
      pred, cost, _ = sess.run([Y_hat, loss, opt], feed_dict={ X: X_batch, Y: Y_batch })

    if i%50 == 0:
      saver.save(sess, './' + test_name+'/training', global_step=i)
      pred_dev = sess.run(Y_hat, feed_dict={ X: X_dev, Y: Y_dev_E })
      pred_dev = np.argmax(pred_dev, axis=1)
      misclassified, class_rate = classification_rate(Y_dev, pred_dev)
      print('--------------------')
      print('---| iter '+str(i))
      print('---| cost '+str(cost))
      print('---| dev class '+str(class_rate))
      print('---| dev misclassified '+str(misclassified))

    if i%250 == 0:
      pred_train_dev, cost, _ = sess.run([Y_hat, loss, opt], feed_dict={ X: X_train_dev, Y: Y_train_dev_E })
      pred_train_dev = np.argmax(pred_train_dev, axis=1)
      misclassified, class_rate = classification_rate(Y_train_dev, pred_train_dev)
      print('---| train-dev class ' + str(class_rate))
      print('---| train-dev misclassified ' + str(misclassified))
  
  pred_test, cost, _ = sess.run([Y_hat, loss, opt], feed_dict={ X: X_test, Y: Y_test_E })
  pred_test = np.argmax(pred_test, axis=1)
  print('class' + str(classification_rate(Y_test, pred_test)))
