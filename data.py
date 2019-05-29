import gzip
import numpy as np
import cv2
import os
from preprocess import process_image
from utils import unison_shuffled_copies

import matplotlib.pyplot as plt
MNIST_train_num = 60000
MNIST_test_num = 10000
image_size = 28

def get_MNIST_X_train(num_train):
  f_train = gzip.open('train-images-idx3-ubyte.gz','r')
  f_train.read(16)
  buf = f_train.read(image_size * image_size * num_train)
  data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
  data = data.reshape(num_train, image_size * image_size)

  return data

def get_MNIST_Y_train(num_train):
  f_labels = gzip.open('train-labels-idx1-ubyte.gz','r')
  f_labels.read(8)
  buf = f_labels.read(num_train)
  labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

  return labels


def get_MNIST_X_test(num_test):
  f_train = gzip.open('t10k-images-idx3-ubyte.gz','r')
  f_train.read(16)
  buf = f_train.read(image_size * image_size * num_test)
  data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
  data = data.reshape(num_test, image_size * image_size)

  return data

def get_MNIST_Y_test(num_test):
  f_labels = gzip.open('t10k-labels-idx1-ubyte.gz','r')
  f_labels.read(8)
  buf = f_labels.read(num_test)
  labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

  return labels

def get_MNIST_train(num):
  X_train = get_MNIST_X_train(num)
  Y_train = get_MNIST_Y_train(num)

  return X_train, Y_train

def get_MNIST_test(num):
  X_test = get_MNIST_X_test(num)
  Y_test = get_MNIST_Y_test(num)

  return X_test, Y_test

def get_all_MNSIT():
  X_train, Y_train = get_MNIST_train(MNIST_train_num)
  X_test, Y_test = get_MNIST_test(MNIST_test_num)

  X = np.concatenate((X_train, X_test))
  Y = np.concatenate((Y_train, Y_test))

  X, Y = unison_shuffled_copies(np.array(X), np.array(Y))

  return X, Y

def get_CANVAS():
  folder = 'canvas_images'
  X = []
  Y = []
  for number in range(0, 10):
    for filename in os.listdir(folder + '/' +str(number)):
      img = cv2.imread(os.path.join(folder, str(number), filename), cv2.IMREAD_GRAYSCALE)
      if img is not None:
        img = process_image(img)
        X.append(img.reshape(image_size * image_size))
        Y.append(number)

  X, Y = unison_shuffled_copies(np.array(X), np.array(Y))
  # split data in train, dev, test set
  m = int(X.shape[0]/3)
  X_canvas_train = X[0:m,:]
  Y_canvas_train = Y[0:m]
  X_dev =  X[m:(2*m),:]
  Y_dev =  Y[m:(2*m)]
  X_test =  X[(2*m):(3*m),:]
  Y_test =  Y[(2*m):(3*m)]

  return X_canvas_train, Y_canvas_train, X_dev, Y_dev, X_test, Y_test

  