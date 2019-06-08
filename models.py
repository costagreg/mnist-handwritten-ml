
def 2_layers_nn(X, output_layer):
  layer_1 = 784
  layer_2 = hidden_layer
  layer_3 = hidden_layer
  layer_4 = output_layer

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

  return Z3


def le_net_5(X, output_layer):
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