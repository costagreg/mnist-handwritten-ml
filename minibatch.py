import gzip
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
np.set_printoptions(threshold=np.inf)

num_train = 600000
num_test = 5
batch_size = 32

def get_X_train():
  f_train = gzip.open('train-images-idx3-ubyte.gz','r')
  image_size = 28
  f_train.read(16)
  buf = f_train.read(image_size * image_size * num_train)
  data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
  data = data.reshape(num_train, image_size * image_size)
  # image = data[2].reshape( image_size, image_size)
  # print(image.shape)
  # plt.imshow(image)
  # plt.show()
  return data

def get_Y_train():
  f_labels = gzip.open('train-labels-idx1-ubyte.gz','r')
  f_labels.read(8)
  buf = f_labels.read(num_train)
  labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  # print(labels[2])
  return labels


def get_X_test():
  f_train = gzip.open('t10k-images-idx3-ubyte.gz','r')
  image_size = 28
  f_train.read(16)
  buf = f_train.read(image_size * image_size * num_test)
  data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
  data = data.reshape(num_test, image_size * image_size)
  # image = np.asarray(data[2262]).squeeze()
  # plt.imshow(image)
  # plt.show()
  return data

def get_Y_test():
  f_labels = gzip.open('t10k-labels-idx1-ubyte.gz','r')
  f_labels.read(8)
  buf = f_labels.read(num_test)
  labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  labels = labels.reshape(num_test, 1)
  # print(labels[2262])
  return labels

def classification_rate(Y, P):
 n_correct = 0
 n_total = 0
 for i in range(len(Y)):
   n_total += 1
   if Y[i] == P[i]:
     n_correct += 1
 return float(n_correct) / n_total

# Get training data
X_train = np.transpose(get_X_train())
Y_train = np.transpose(get_Y_train())

# Normalize data
X_train = X_train/255.

Y_train_E = np.eye(10)[Y_train] #hot encoding
Y_train_E = np.transpose(Y_train_E)

print('X_train shape ' + str(X_train.shape))
print('Y_train shape ' + str(Y_train_E.shape))

# Get test data
X_test = get_X_test()
Y_test = get_Y_test()

print('X_test shape ' + str(X_test.shape))
print('Y_test shape ' + str(Y_test.shape))

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

# saver = tf.train.Saver()
# tf.reset_default_graph()
# imported_graph = tf.train.import_meta_graph('./my_test_model-12000.meta')

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # imported_graph.restore(sess, './my_test_model-12000')
  # X_train = create_batches(X_train)
  # (num_batches, layer_1, m)
  num_batches = int(np.ceil(X_train.shape[1]/batch_size))
  for i in range(10000):
    for j in range(num_batches):
      batch_start = batch_size*j
      batch_end = batch_size*(j+1)
      X_batch =  X_train[:, batch_start:batch_end]
      Y_batch =  Y_train_E[:, batch_start:batch_end]
      pred, cost, _ = sess.run([Y_hat, loss, opt], feed_dict={ X: X_batch, Y: Y_batch })

    if i%1000 == 0:
      print(cost)
  
  pred, cost, _ = sess.run([Y_hat, loss, opt], feed_dict={ X: X_train, Y: Y_train_E })
  pred = np.argmax(pred, axis=0)
  print(pred)
  print(Y_train)
  print(classification_rate(Y_train, pred))