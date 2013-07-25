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
    def __init__(self, shape, nonlinearity=np.tanh, nonlinearity_deriv=tanh_deriv, init_weight_scale=.05):
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
            self.weights.append(init_weight_scale*np.random.randn(self.shape[i],self.shape[i+1]))
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
        pass

    def computeDelta(self, layer_deltas, weights, node_activations, d_nonlinearity):
        """
        Compute error and delta for layer the weights project to
        """
        delta_error = np.sum(weights * layer_deltas, axis=1)
        return delta_error * d_nonlinearity(node_activations)

    def adjustWeights(self, layer_deltas, node_activations, weights, c):
        """
        Adjust given weights based on layer deltas
        """
        change =  layer_deltas * node_activations
        weights += self.learning_rate * change + self.momentum * c
        c = change

    def adjust_weights_to_outlayer(self, output_error, outlayer, hiddenlayer, weights, c, d_nonlinearity):
        """
        Adjust weights to the output layer of a network
        """
        # Compute output layer delta
        layer_deltas = output_error * d_nonlinearity(outlayer)
        # Modify weights to output layer
        self.adjustWeights(layer_deltas[np.newaxis, :], hiddenlayer, weights, c)
        return layer_deltas

    def adjust_weights_to_hiddenlayer(self, layer_deltas, postlayer_weights, inputlayer, hiddenlayer, weights, c,
                                      d_nonlinearity):
        """
        Adjust weights to the hidden layer of a network
        """
        layer_deltas = self.computeDelta(layer_deltas, postlayer_weights, hiddenlayer, d_nonlinearity)
        # Compute change in weights
        self.adjustWeights(layer_deltas[np.newaxis, :],inputlayer, weights, c)

    def backPropagate(self, net, output_error):
        """
        Back propagate error
        """

        # Adjust weights to output layer
        outlayer=net.node_activations[-1]
        hiddenlayer=net.node_activations[-2]
        layer_deltas = self.adjust_weights_to_outlayer(output_error, outlayer,  hiddenlayer[:, np.newaxis],
            net.weights[-1], net.c[-1], net.nonlinearity_deriv)

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
                self.adjust_weights_to_hiddenlayer(layer_deltas, postlayer_weights,inputlayer[:, np.newaxis],
                    hiddenlayer, net.weights[weight_idx], net.c[weight_idx], net.nonlinearity_deriv)

        # calc combined error
        return layer_deltas

    def test(self, net, in_patterns, out_patterns):
        """
        Test network on given patterns
        """
        for i in range(len(in_patterns)):
            inputs = in_patterns[i]
            print 'Inputs:', inputs, '-->', net.run(inputs), '\tTarget', out_patterns[i]

    def train(self, net, in_patterns, out_patterns, max_iterations = 100000, err_thresh=.0001):
        recent_error=np.zeros([100])
        for i in range(max_iterations):
            j=np.random.randint(0,high=len(in_patterns))
            inputs=in_patterns[j]
            targets=out_patterns[j]
            # Run network with pattern input
            output=net.run(inputs)

            # Compute output layer error
            output_error = self.error_deriv(targets, output)
            # back propagate
            self.backPropagate(net, output_error)
            pat_error =np.sum(self.error(targets,output))
            recent_error[0:-1]=recent_error[1:]
            recent_error[-1]=pat_error
            if i>=100:
                total_error=np.sum(recent_error)
                if i % 50 == 0:
                    print 'Combined error', total_error

                # Quit if converged
                if i>0 and total_error<err_thresh:
                    break
        # Test network
        self.test(net, in_patterns, out_patterns)

def test_xor():
    in_pat = [[0,0],[0,1],[1,0],[1,1]]
    out_pat = [[0],[1],[1],[0]]
    myNN = FeedforwardNetwork([2, 3, 2, 1])
    trainer = BackPropTrainer()
    trainer.train(myNN,in_pat,out_pat)
    myNN.print_weights()

def test_autoencode():
    input_patterns=[[0,0],[0,1],[1,0],[1,1]]
    in_patterns=[]
    out_patterns=[]
    for i in range(len(input_patterns)):
        for j in range(50):
            in_patterns.append(input_patterns[i]+.1*np.random.randn(len(input_patterns[i])))
            out_patterns.append(input_patterns[i])
    myNN=FeedforwardNetwork([2,10,2])
    trainer = BackPropTrainer(learning_rate=0.1, momentum=0.0)
    trainer.train(myNN,in_patterns,out_patterns,err_thresh=0.001)
    myNN.print_weights()

if __name__ == "__main__":
    test_autoencode()
    #test_xor()