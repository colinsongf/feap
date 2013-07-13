import numpy as np

class Network(object):
    """
        Represents a layered artificial neural network.
    """

    def __init__(self, shape, num_inputs):
        """
            Construct a Network instance.

            shape: an array of length L, where L is the number of layers, including the input layer. Each element
                   contains the number of neurons in that layer. shape[0] should be equal to the number of inputs.
        """
        self.shape = shape
        self.num_layers = len(self.shape)
        self.num_inputs = self.shape[0]
        self.weights = self.create_weight_matrix(self.shape)
        self.input_weights = self.create_input_weight_matrix(self.shape)

    def create_weight_matrix(self, shape):
        """
            Create an N x N weight matrix.
        """
        L = shape[0]
        N = np.sum(shape[1:]) #total number of neurons across layers
        raise NotImplementedError('Not implemented yet!')

    def create_input_weight_matrix(self, shape):
        """
            Create an N x num_inputs weight matrix.
        """
        raise NotImplementedError('Not implemented yet!')

    def connect_input(self, input_index, i, weight):
        """
            Connect an input from neuron i to an element of the input.

            i: the neuron number
            input_index: the index of the element of the input to neuron i
            weight: the input weight
        """
        self.input_weights[i, input_index] = weight

    def connect_neuron(self, i, j, weight):
        """
            Create a connection from neuron i to neuron j with the given weight.
        """
        self.weights[i, j] = weight
















