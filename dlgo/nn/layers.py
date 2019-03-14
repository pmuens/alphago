import numpy as np

class Layer:
    def __init__(self):
        self.params = []

        self.previous = None
        self.next = None

        # state for the forward pass
        self.input_data = None
        self.output_data = None

        # state for the backward pass
        self.input_delta = None
        self.output_delta = None

    def connect(self, layer):
        self.previous = layer
        layer.next = self

    def forward(self):
        raise NotImplementedError

    def get_forward_input(self):
        if self.previous is not None:
            return self.previous.output_data
        else:
            return self.input_data

    def backward(self):
        raise NotImplementedError

    def get_backward_input(self):
        if self.next is not None:
            return self.next.output_delta
        else:
            return self.input_delta

    def clear_deltas(self):
        pass

    def update_params(self, learning_rate):
        pass

    def describe(self):
        raise NotImplementedError

class ActivationLayer(Layer):
    def __init__(self, input_dim):
        super(ActivationLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self):
        data = self.get_forward_input()
        self.output_data = sigmoid(data)

    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        self.output_delta = delta * sigmoid_prime(data)

    def describe(self):
        print("|-- " + self.__class__.__name__)
        print("  |-- dimensions: ({}, {})"
            .format(self.input_dim, self.output_dim))

class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = np.random.randn(output_dim, input_dim)
        self.bias = np.random.randn(output_dim, 1)

        self.params = [self.weight, self.bias]

        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def forward(self):
        data = self.get_forward_input()
        self.output_data = np.dot(self.weight, data) + self.bias

    def backward(self):
        data = self.get_forward_input()
        delta = self.get_backward_input()

        self.delta_b += delta
        self.delta_w += np.dot(delta, data.transpose())
        self.output_delta = np.dot(self.weight.transpose(), delta)

    def update_params(self, rate):
        self.weight -= rate * self.delta_w
        self.bias -= rate * self.delta_b

    def clear_deltas(self):
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def describe(self):
        print("|-- " + self.__class__.__name__)
        print("  |-- dimensions: ({}, {})"
            .format(self.input_dim, self.output_dim))

# computes sigmoid on doubles
def sigmoid_double(x):
    return 1.0 / (1.0 + np.exp(-x))

# computes sigmoid on vectors
def sigmoid(z):
    return np.vectorize(sigmoid_double)(z)

# derivative of sigmoid function
def sigmoid_prime_double(x):
    return sigmoid_double(x) * (1 - sigmoid_double(x))

def sigmoid_prime(z):
    return np.vectorize(sigmoid_prime_double)(z)
