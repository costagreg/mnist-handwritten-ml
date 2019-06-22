import gzip
import numpy as np
import cv2
import os
from preprocess import process_image
from utils import unison_shuffled_copies, prepare_Y, prepare_X

import matplotlib.pyplot as plt
MNIST_train_num = 60000
MNIST_test_num = 10000
image_size = 28


def get_MNIST_X_train(num_train):
    f_train = gzip.open('train-images-idx3-ubyte.gz', 'r')
    f_train.read(16)
    buf = f_train.read(image_size * image_size * num_train)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_train, image_size, image_size, 1)

    return data


def get_MNIST_Y_train(num_train):
    f_labels = gzip.open('train-labels-idx1-ubyte.gz', 'r')
    f_labels.read(8)
    buf = f_labels.read(num_train)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

    return labels


def get_MNIST_X_test(num_test):
    f_train = gzip.open('t10k-images-idx3-ubyte.gz', 'r')
    f_train.read(16)
    buf = f_train.read(image_size * image_size * num_test)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_test, image_size, image_size, 1)

    return data


def get_MNIST_Y_test(num_test):
    f_labels = gzip.open('t10k-labels-idx1-ubyte.gz', 'r')
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

    X = np.pad(X, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X, Y = unison_shuffled_copies(np.array(X), np.array(Y))

    return X, Y


def get_CANVAS():
    folder = 'canvas_images'
    X = []
    Y = []
    for number in range(0, 10):
        for filename in os.listdir(folder + '/' + str(number)):
            img = cv2.imread(os.path.join(folder, str(number),
                                          filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = process_image(img)
                # TODO: is this reshape needed?
                X.append(img.reshape(image_size, image_size, 1))
                Y.append(number)

    X = np.pad(X, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X, Y = unison_shuffled_copies(np.array(X), np.array(Y))
    # split data in train, dev, test set
    train_size = int(X.shape[0]/2)
    dev_size = int(X.shape[0]/4)
    test_size = dev_size

    X_canvas_train = X[0:train_size, :]
    Y_canvas_train = Y[0:train_size]
    X_dev = X[train_size:(train_size+dev_size), :]
    Y_dev = Y[train_size:(train_size+dev_size)]
    X_test = X[train_size+dev_size:, :]
    Y_test = Y[train_size+dev_size:]

    return X_canvas_train, Y_canvas_train, X_dev, Y_dev, X_test, Y_test


def get_data(print_shape=False):
    # Get data from MINST and CANVAS data
    X_minst, Y_minst = get_all_MNSIT()
    X_canvas_train, Y_canvas_train, X_dev, Y_dev, X_test, Y_test = get_CANVAS()

    # Split data between train and train-dev
    X_train = np.concatenate((X_minst, X_canvas_train))
    Y_train = np.concatenate((Y_minst, Y_canvas_train))
    X_train, Y_train = unison_shuffled_copies(X_train, Y_train)
    X_train_dev = X_train[0:X_dev.shape[0], :]
    Y_train_dev = Y_train[0:X_dev.shape[0]]
    X_train = X_train[X_dev.shape[0]:, :]
    Y_train = Y_train[X_dev.shape[0]:]

    # Prepare training data and training-dev data
    X_train = prepare_X(X_train)
    Y_train, Y_train_E = prepare_Y(Y_train, 10)
    X_train_dev = prepare_X(X_train_dev)
    Y_train_dev, Y_train_dev_E = prepare_Y(Y_train_dev, 10)

    # Prepare test data
    X_test = prepare_X(X_test)
    Y_test, Y_test_E = prepare_Y(Y_test, 10)

    # Prepare dev data
    X_dev = prepare_X(X_dev)
    Y_dev, Y_dev_E = prepare_Y(Y_dev, 10)

    if print_shape == True:
        print('X_train shape ' + str(X_train.shape))
        print('Y_train shape ' + str(Y_train_E.shape))
        print('X_train_dev shape ' + str(X_train_dev.shape))
        print('Y_train_dev shape ' + str(Y_train_dev_E.shape))
        print('X_test shape ' + str(X_test.shape))
        print('Y_test shape ' + str(Y_test_E.shape))

    return {
        'X_train': X_train,
        'Y_train': Y_train,
        'Y_train_E': Y_train_E,
        'X_train_dev': X_train_dev,
        'Y_train_dev': Y_train_dev,
        'Y_train_dev_E': Y_train_dev_E,
        'X_test': X_test,
        'Y_test': Y_test,
        'Y_test_E': Y_test_E,
        'X_dev': X_dev,
        'Y_dev': Y_dev,
        'Y_dev_E': Y_dev_E
    }
