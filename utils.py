import numpy as np 

def hot_encoding(data, num_label):
  return np.transpose(np.eye(num_label)[data])

def classification_rate(Y, P):
 n_correct = 0
 n_total = 0
 for i in range(len(Y)):
   n_total += 1
   if Y[i] == P[i]:
     n_correct += 1
 return float(n_correct) / n_total

def prepare_X(data):
  X = np.transpose(data)
  X = X / 255.
  return X

def prepare_Y(data, hot_encoding_labels):
  Y = np.transpose(data)
  Y_E = hot_encoding(Y, hot_encoding_labels)

  return Y, Y_E

def ValueInvert(array):
    # Flatten the array for looping
    flatarray = array.flatten()
    
    # Apply transformation to flattened array
    for i in range(flatarray.size):
        flatarray[i] = 255 - flatarray[i]
        
    # Return the transformed array, with the original shape
    return flatarray.reshape(array.shape)