import numpy as np 
import sys

def hot_encoding(data, num_label):
  return np.eye(num_label)[data]

def classification_rate(Y, P):
 n_correct = 0
 n_total = 0
 misclassified = {
   '0':0,
   '1':0,
   '2':0,
   '3':0,
   '4':0,
   '5':0,
   '6':0,
   '7':0,
   '8':0,
   '8':0,
   '9':0
 }
 for i in range(len(Y)):
   n_total += 1
   if Y[i] == P[i]:
     n_correct += 1
   else:
     misclassified[str(Y[i])] += 1
 return misclassified, float(n_correct) / n_total

def prepare_X(data):
  # X = np.transpose(data)
  X = data / 255.
  return X

def prepare_Y(data, hot_encoding_labels):
  # Y = np.transpose(data)
  Y_E = hot_encoding(data, hot_encoding_labels)

  return data, Y_E

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def ValueInvert(array):
    # Flatten the array for looping
    flatarray = array.flatten()
    
    # Apply transformation to flattened array
    for i in range(flatarray.size):
        flatarray[i] = 255 - flatarray[i]
        
    # Return the transformed array, with the original shape
    return flatarray.reshape(array.shape)

def read_variable_from_batch():
  # test_name, num_hidden_layer, learning rate
  return str(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
