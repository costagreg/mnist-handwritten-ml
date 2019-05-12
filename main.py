import gzip
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_train = 60000 
num_test = 10000

def get_X_train():
  f_train = gzip.open('train-images-idx3-ubyte.gz','r')
  image_size = 28
  f_train.read(16)
  buf = f_train.read(image_size * image_size * num_train)
  data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
  data = data.reshape(num_train, image_size, image_size)
  # image = np.asarray(data[2]).squeeze()
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
  data = data.reshape(num_test, image_size, image_size)
  # image = np.asarray(data[2262]).squeeze()
  # plt.imshow(image)
  # plt.show()
  return data

def get_Y_test():
  f_labels = gzip.open('t10k-labels-idx1-ubyte.gz','r')
  f_labels.read(8)
  buf = f_labels.read(num_test)
  labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  # print(labels[2262])
  return labels

# Get training data
X_train = get_X_train()
Y_train = get_Y_train()

print('X_train shape ' + str(X_train.shape))
print('Y_train shape ' + str(Y_train.shape))

# Get test data
X_test = get_X_test()
Y_test = get_Y_test()

print('X_test shape ' + str(X_test.shape))
print('Y_test shape ' + str(Y_test.shape))

# 2 layer  NN

layer_1 = 36
layer_2 = 20
layer_3 = 10

