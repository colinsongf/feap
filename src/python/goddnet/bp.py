import numpy as np

def tanh_deriv(x):
    return 1-np.tanh(x)**2

def half_sq_error(x1,x2):
    return 0.5*(x1-x2)**2

def half_sq_error_deriv(x1,x2):
    return x1-x2

class FeedforwardNetwork():
    """
    Represents a feedforward neural network
    """
    def __init__(self, shape, nonlinearity=np.tanh, nonlinearity_deriv=tanh_deriv):
        self.shape=shape
        self.nonlinearity=nonlinearity
        self.nonlinearity_deriv=nonlinearity_deriv
        self.n_layers=len(shape)
        #bias
        self.shape[0]+=1
        self.weights=[]
        self.c=[]
        # Init weights and weight change
        for i in range(len(self.shape)-1):
            self.weights.append(.05*np.random.randn(self.shape[i],self.shape[i+1]))
            self.c.append(np.zeros([self.shape[i],self.shape[i+1]]))
        # Init layer nodes
        self.node_activations=[]
        for i in range(len(self.shape)):
            self.node_activations.append(np.ones([self.shape[i]]))

    def run (self, inputs):
        """
        Run input through the network
        """
        if len(inputs) != self.shape[0]-1:
            print 'incorrect number of inputs'

        # Set input and bias
        self.node_activations[0][:-1]=inputs[:]
        self.node_activations[0][-1]=1.0

        # Propagate activity through layers
        for i in range(self.n_layers-1):
            self.node_activations[i+1]=self.nonlinearity(np.sum(self.node_activations[i][:,np.newaxis]*self.weights[i],
                axis=0))

        # Return output layer activity
        return self.node_activations[-1]

    def print_weights(self):
        """
        Print all weight values
        """
        for i in range(len(self.shape)-1):
            print'Layer %d -> Layer %d weights' % (i,i+1)
            for j in range(self.shape[i]):
                print self.weights[i][j,:]
            print ''


class BackPropTrainer():
    """
    Trains a feedforward network using backprop
    """
    def __init__(self, learning_rate=0.1, momentum=0.05, error=half_sq_error, error_deriv=half_sq_error_deriv):
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.error=error
        self.error_deriv=error_deriv

    def train (self, net, patterns, max_iterations = 100000, err_thresh=.001):
        """
        Train on the given patterns
        """
        for i in range(max_iterations):
            # total error over all patterns
            total_error=0
            for p in patterns:
                inputs = p[0]
                targets = p[1]

                # Run network with pattern input
                net.run(inputs)

                # back propagate
                (error,layer_deltas)=self.backPropagate(net, targets)

                total_error +=error

            if i % 50 == 0:
                print 'Combined error', total_error

            # Quit if converged
            if i>0 and total_error<err_thresh:
                break

        # Test network
        self.test(net, patterns)

    def computeDelta(self, layer_deltas, weights, node_activations, d_nonlinearity):
        """
        Compute error and delta for layer the weights project to
        """
        delta_error = np.sum(weights * layer_deltas, axis=1)
        layer_deltas = delta_error * d_nonlinearity(node_activations)
        return layer_deltas

    def adjustWeights(self, layer_deltas, node_activations, weights, c):
        """
        Adjust given weights based on layer deltas
        """
        change =  layer_deltas * node_activations
        weights += self.learning_rate * change + self.momentum * c
        c = change
        return weights,c

    def adjust_weights_to_outlayer(self, targets, outlayer, hiddenlayer, weights, c, d_nonlinearity):
        """
        Adjust weights to the output layer of a network
        """
        # Compute output layer error
        output_error = self.error_deriv(targets, outlayer)
        # Compute output layer delta
        layer_deltas = output_error * d_nonlinearity(outlayer)
        # Modify weights to output layer
        (weights, c) = self.adjustWeights(layer_deltas[np.newaxis, :], hiddenlayer, weights, c)
        return layer_deltas,weights,c

    def adjust_weights_to_hiddenlayer(self, layer_deltas, postlayer_weights, inputlayer, hiddenlayer, weights, c,
                                      d_nonlinearity):
        """
        Adjust weights to the hidden layer of a network
        """
        layer_deltas = self.computeDelta(layer_deltas, postlayer_weights,
            hiddenlayer, d_nonlinearity)
        # Compute change in weights
        (weights, c) = self.adjustWeights(layer_deltas[np.newaxis, :],inputlayer, weights, c)
        return layer_deltas,weights,c

    def backPropagate(self, net, targets):
        """
        Back propagate error
        """

        # Adjust weights to output layer
        outlayer=net.node_activations[-1]
        hiddenlayer=net.node_activations[-2]
        (layer_deltas,net.weights[-1],net.c[-1]) = self.adjust_weights_to_outlayer(targets, outlayer,
            hiddenlayer[:, np.newaxis], net.weights[-1], net.c[-1], net.nonlinearity_deriv)

        # If more than two layer network, modify remaining weights, moving back a layer at a time (starting at weights
        # to hidden layer just before output layer)
        if net.n_layers>2:
            for weight_idx in range(net.n_layers-3,-1,-1):
                # Index of layer that this weight matrix projects to
                layer_idx=weight_idx+1

                # Adjust weights to hidden layer
                postlayer_weights=net.weights[weight_idx + 1]
                inputlayer=net.node_activations[weight_idx]
                hiddenlayer=net.node_activations[layer_idx]
                (layer_deltas,net.weights[weight_idx], net.c[weight_idx])=self.adjust_weights_to_hiddenlayer(layer_deltas,
                    postlayer_weights,inputlayer[:, np.newaxis], hiddenlayer, net.weights[weight_idx], net.c[weight_idx],
                    net.nonlinearity_deriv)

        # calc combined error
        return np.sum(self.error(targets,net.node_activations[-1])),layer_deltas

    def test(self, net, patterns):
        """
        Test network on given patterns
        """
        for p in patterns:
            inputs = p[0]
            print 'Inputs:', p[0], '-->', net.run(inputs), '\tTarget', p[1]

def test_xor():
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]
    myNN = FeedforwardNetwork([2, 3, 2, 1])
    trainer = BackPropTrainer()
    trainer.train(myNN,pat)
    myNN.print_weights()

def test_autoencode():
    input_patterns=[[0,0],[0,1],[1,0],[1,1]]
    patterns=[]
    for i in range(len(input_patterns)):
        for j in range(50):
            test_input=input_patterns[i]+.1*np.random.randn(len(input_patterns[i]))
            test_output=input_patterns[i]
            patterns.append([test_input,test_output])
    myNN=FeedforwardNetwork([2,10,2])
    trainer = BackPropTrainer(learning_rate=0.1, momentum=0.0)
    trainer.train(myNN,patterns,err_thresh=0.01)
    myNN.print_weights()

if __name__ == "__main__":
    test_autoencode()
    #test_xor()