import numpy as np

def tanh_deriv(x):
    return 1-np.tanh(x)**2

class FeedforwardNetwork():
    def __init__(self, shape, nonlinearity=np.tanh, nonlinearity_deriv=tanh_deriv):
        self.shape=shape
        self.nonlinearity=nonlinearity
        self.nonlinearity_deriv=nonlinearity_deriv
        self.n_layers=len(shape)
        self.shape[0]+=1 #bias
        self.weights=[]
        self.c=[]
        for i in range(len(self.shape)-1):
            self.weights.append(.05*np.random.randn(self.shape[i],self.shape[i+1]))
            self.c.append(np.zeros([self.shape[i],self.shape[i+1]]))
        self.node_activations=[]
        for i in range(len(self.shape)):
            self.node_activations.append(np.ones([self.shape[i]]))

    def run (self, inputs):
        if len(inputs) != self.shape[0]-1:
            print 'incorrect number of inputs'

        self.node_activations[0][:-1]=inputs[:]

        for i in range(self.n_layers-1):
            sum=np.sum(self.node_activations[i][:,np.newaxis]*self.weights[i],axis=0)
            self.node_activations[i+1]=self.nonlinearity(sum)

        return self.node_activations[-1]

    def print_weights(self):
        for i in range(len(self.shape)-1):
            print'Layer %d -> Layer %d weights' % (i,i+1)
            for j in range(self.shape[i]):
                print self.weights[i][j,:]
            print ''

class BackPropTrainer():
    def __init__(self, learning_rate=0.1, momentum=0.05):
        self.learning_rate=learning_rate
        self.momentum=momentum

    def train (self, net, patterns, max_iterations = 100000):
        for i in range(max_iterations):
            error=0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                net.run(inputs)
                error += self.backPropagate(net, targets)
            if i % 50 == 0:
                print 'Combined error', error
            if i>0 and error<.001:
                break
        self.test(net, patterns)

    def backPropagate(self, net, targets):

        # Compute output layer error
        error=targets-net.node_activations[-1]
        # Compute output layer delta
        layer_deltas=error*net.nonlinearity_deriv(net.node_activations[-1])

        # Modify weights to output layer
        change=layer_deltas[np.newaxis,:]*net.node_activations[-2][:,np.newaxis]
        net.weights[-1]+=self.learning_rate*change+self.momentum*net.c[-1]
        net.c[-1]=change

        # If more than two layer network, modify remaining weights, moving back a layer at a time (starting at weights
        # to hidden layer just before output layer)
        if net.n_layers>2:
            for weight_idx in range(net.n_layers-3,-1,-1):
                # Index of layer that this weight matrix projects to
                layer_idx=weight_idx+1

                # Compute error and delta for layer the weights project to
                error=np.sum(net.weights[weight_idx+1]*layer_deltas,axis=1)
                layer_deltas=error*net.nonlinearity_deriv(net.node_activations[layer_idx])

                # Compute change in weights
                change=layer_deltas[np.newaxis,:]*net.node_activations[weight_idx][:,np.newaxis]
                net.weights[weight_idx]+=self.learning_rate*change+self.momentum*net.c[weight_idx]
                net.c[weight_idx]=change

        # calc combined error
        return 0.5*np.sum((targets-net.node_activations[-1])**2)

    def test(self, net, patterns):
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
    trainer = BackPropTrainer(learning_rate=0.2, momentum=0.0)
    trainer.train(myNN,patterns)
    myNN.print_weights()

if __name__ == "__main__":
    test_autoencode()