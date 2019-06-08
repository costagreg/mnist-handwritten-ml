import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

from data import  get_data
from utils import classification_rate

def(X, output_layer):
  C1_w = tf.Variable(tf.truncated_normal(shape=[5,5,1,6], mean=0, sigma=.1))
  C1_b = tf.Variable(tf.zeros(6))
  C1 = tf.nn.conv2d(X, C1_w, strides=[1,1,1,1], padding='VALID', name='C1') + C1_b

  P2 = tf.nn.avg_pool(C1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='P2')

  C3_w = tf.Variable(tf.truncated_normal(shape=[5,5,6,16]), mean=0, sigma=.1))
  C3_b = tf.Variable(tf.zeros(16))
  C3 = tf.mm.conv2d(P2, C3_w, strides=[1,1,1,1], padding='VALID', name='C3') + C3_b

  P4 = tf.nn.avg_pool(C3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID', name='P4')

  F5 = tensorflow.contrib.layers.flatten(P4)
  F5_w = tf.Variable(tf.truncated_normal(shape=[400, 120], mean=0, sigma=.1))
  F5_b = tf.Variable(tf.zeros(120))
  F5 = tf.matmul(F5, F5_w) + F5_b
  F5 = tf.nn.relu(F5)

  F6_w = tf.Variable(tf.truncated_norma(shape=[120, 84], mean=0, sigm=0.1))
  F6_b = tf.Variable(tf.zeros(84))
  F6 = tf.matmul(F5, F6_w) + F6_b

  F7_w = tf.Variable(tf.truncated_norma(shape=[84, output_layer], mean=0, sigm=0.1))
  F7_b = tf.Variable(tf.zeros(output_layer))

  Y_hat = tf.matmul(F6, F7_w) + F7_b

  return Y_hat