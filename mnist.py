import gzip
import numpy as np

image_size = 28

def get_X_train(num_train):
  f_train = gzip.open('train-images-idx3-ubyte.gz','r')
  f_train.read(16)
  buf = f_train.read(image_size * image_size * num_train)
  data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
  data = data.reshape(num_train, image_size * image_size)

  return data

def get_Y_train(num_train):
  f_labels = gzip.open('train-labels-idx1-ubyte.gz','r')
  f_labels.read(8)
  buf = f_labels.read(num_train)
  labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

  return labels


def get_X_test(num_test):
  f_train = gzip.open('t10k-images-idx3-ubyte.gz','r')
  f_train.read(16)
  buf = f_train.read(image_size * image_size * num_test)
  data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
  data = data.reshape(num_test, image_size * image_size)

  return data

def get_Y_test(num_test):
  f_labels = gzip.open('t10k-labels-idx1-ubyte.gz','r')
  f_labels.read(8)
  buf = f_labels.read(num_test)
  labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

  return labels

def get_train(num):
  X_train = get_X_train(num)
  Y_train = get_Y_train(num)

  return X_train, Y_train


def get_test(num):
  X_test = get_X_test(num)
  Y_test = get_Y_test(num)

  return X_test, Y_test

if __name__ == "__main__":
    print(get_X_train(1))
    X = np.transpose(get_X_train(1))
    X = X / 255.
    print(X)