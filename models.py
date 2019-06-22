import tensorflow as tf


def le_net_5(X, output_layer):

    # CONV1
    C1_w = tf.Variable(tf.truncated_normal(
        shape=[5, 5, 1, 6], mean=0, stddev=.1))
    C1_b = tf.Variable(tf.zeros(6))
    C1 = tf.nn.conv2d(X, C1_w, strides=[
                      1, 1, 1, 1], padding='VALID', name='C1') + C1_b
    C1 = tf.nn.relu(C1)

    # POOL2
    P2 = tf.nn.max_pool(C1, ksize=[1, 2, 2, 1], strides=[
                        1, 2, 2, 1], padding='VALID', name='P2')
    # CONV3
    C3_w = tf.Variable(tf.truncated_normal(
        shape=[5, 5, 6, 16], mean=0, stddev=.1))
    C3_b = tf.Variable(tf.zeros(16))
    C3 = tf.nn.conv2d(P2, C3_w, strides=[
                      1, 1, 1, 1], padding='VALID', name='C3') + C3_b
    C3 = tf.nn.relu(C3)

    # POOL4
    P4 = tf.nn.max_pool(C3, ksize=[1, 2, 2, 1], strides=[
                        1, 2, 2, 1], padding='VALID', name='P4')

    # FULL_CONNECT5
    F5 = tf.contrib.layers.flatten(P4)
    F5_w = tf.Variable(tf.truncated_normal(
        shape=[400, 120], mean=0, stddev=.1))
    F5_b = tf.Variable(tf.zeros(120))
    F5 = tf.matmul(F5, F5_w) + F5_b
    F5 = tf.nn.relu(F5)

    # FULL_CONNECT6
    F6_w = tf.Variable(tf.truncated_normal(
        shape=[120, 84], mean=0, stddev=0.1))
    F6_b = tf.Variable(tf.zeros(84))
    F6 = tf.matmul(F5, F6_w) + F6_b
    F6 = tf.nn.relu(F6)

    # FULL_CONNECT7
    F7_w = tf.Variable(tf.truncated_normal(
        shape=[84, output_layer], mean=0, stddev=0.1))
    F7_b = tf.Variable(tf.zeros(output_layer))

    # PRED
    Y_hat = tf.matmul(F6, F7_w) + F7_b

    return Y_hat
