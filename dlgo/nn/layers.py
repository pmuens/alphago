import numpy as np

# computes sigmoid on doubles
def sigmoid_double(x):
    return 1.0 / (1.0 + np.exp(-x))

# computes sigmoid on vectors
def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)
