import numpy as np

# For softmax, Z should be a (# of classes, m) matrix
# Recall that axis=0 is column sum, while axis=1 is row sum
def softmax(Z):
    t = np.exp(Z)
    t = t / t.sum(axis=0, keepdims=True)
    return t

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    return A