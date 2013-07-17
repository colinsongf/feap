import numpy as np
from scipy import sparse

import networkx as nx

def tanh_deriv(x):
    return 1-np.tanh(x)**2

class Network(object):
    """
        Represents a layered artificial neural network.
    """

    def __init__(self, shape, nonlinearity=np.tanh, nonlinearity_deriv=tanh_deriv):
        """
            Construct a Network instance.

            shape: an array of length L, where L is the number of layers, including the input layer. Each element
                   contains the number of neurons in that layer. shape[0] should be equal to the number of inputs.
            nonlinearity: the output nonlinearity of the neuron, defaults to np.tanh. The function must take an
                array of floats as an argument and return an array of the same length with the nonlinearity applied.
        """

        self.shape = shape
        self.num_layers = len(self.shape)
        self.num_inputs = self.shape[0]
        self.nonlinearity = nonlinearity
        self.nonlinearity_deriv = nonlinearity_deriv
        self.compiled = False

        self.graph = self.create_graph(self.shape)

    def create_graph(self, shape):
        """
            Create an unconnected graph with the number of neurons specified by shape.

            shape: an array of length L, where L is the number of layers, including the input layer. Each element
                contains the number of neurons in that layer. shape[0] should be equal to the number of inputs.
        """
        graph = nx.DiGraph()
        neuron_id = 0
        for layer,num_neurons in enumerate(shape):
            for n in range(num_neurons):
                graph.add_node(neuron_id, layer=layer, index=n)
                neuron_id += 1
        return graph

    def connect(self, n1, n2, weight):
        """
            Connect neuron n1 to neuron n2.

            n1: the id of neuron 1
            n2: the id of neuron 2
            weight: the connection weight
        """
        self.graph.add_edge(n1, n2, weight=weight)

    def compile(self):
        """
            Create the numerical objects necessary to simulate and train the network.
        """

        #create state vector
        N = self.graph.number_of_nodes()
        self.state = np.zeros(N, dtype='float')

        self.node2index = dict()
        self.index2node = dict()
        for k,n in enumerate(self.graph.nodes()):
            self.node2index[n] = k
            self.index2node[k] = n

        #identify input nodes
        inodes = [n for n in self.graph.nodes() if self.graph.node[n]['layer'] == 0]
        inodes.sort()
        self.input2index = dict()
        for k,inode in enumerate(inodes):
            self.input2index[inode] = k
        self.input_indices = [self.node2index[n] for n in inodes] #the indices in the state vector corresponding to the input layer

        #identify output nodes
        onodes = [n for n in self.graph.nodes() if self.graph.node[n]['layer']==(self.num_layers-1)]
        onodes.sort()
        self.output2index=dict()
        for k,onode in enumerate(onodes):
            self.output2index[onode]=k
        self.output_indices = [self.node2index[n] for n in onodes]

        #build weight matrix, element ij of the matrix is the weight from node j to node i
        W = sparse.lil_matrix( (N, N), dtype='float')
        for n1,n2 in self.graph.edges():
            j = self.node2index[n1]
            i = self.node2index[n2]
            W[i, j] = self.graph[n1][n2]['weight']
        self.weights = W.tocsr() #more efficient for multiplication

        self.compiled = True

    def step(self, input=None):
        """
            Step the network forward one step.
        """
        if not self.compiled:
            self.compile()
        if input is None:
            self.state[self.input_indices] = 0.0
        else:
            if len(self.input_indices) != len(input):
                print 'Input size mismatch, input layer has %d neurons, input given is of length %d' % (len(self.input_indices), len(input))
            self.state[self.input_indices] = input

        self.state = self.nonlinearity(self.weights*self.state)


class DenseFeedforwardConnector(object):
    """
        Creates dense feedforward connections between layers of a network.
    """

    def __init__(self, random=False):
        self.random = random

    def connect(self, net):
        for k in range(net.num_layers-1):
            pre_layer = [n for n in net.graph.nodes() if net.graph.node[n]['layer'] == k]
            post_layer = [n for n in net.graph.nodes() if net.graph.node[n]['layer'] == k+1]
            for n1 in pre_layer:
                for n2 in post_layer:
                    w = 0.0
                    if self.random:
                        w = np.random.randn()
                    net.connect(n1, n2, weight=w)


class Trainer(object):

    def __init__(self):
        pass

    def get_updates(self, net, sample_input):
        """
            Computes the change in weights for the network given a sample input.

            net: the network to update
            sample_input: the training example

            Should return a dictionary { (n11, n12):delta_w1, ..., (ni1, ni2):delta_wi } where the key is the edge
            and the value is the change in weight for that connection.
        """
        delta_W = dict()
        for n1,n2 in net.graph.edges():
            delta_W[(n1, n2)] = 0.0
        return delta_W

class BackPropTrainer(Trainer):

    def __init__(self, alpha):
        Trainer.__init__(self)
        self.alpha=alpha

    def train(self, net, sample_input, target_output):
        delta_W=self.get_updates(self, net, sample_input, target_output)
        net.weights-=self.alpha*delta_W

class OnlineBackPropTrainer(BackPropTrainer):

    def __init__(self, alpha):
        BackPropTrainer.__init__(self, alpha)

    def train(self, net, sample_input, target_output):
        net.step(input=sample_input)

        layer_nodes = [n for n in net.graph.nodes() if net.graph.node[n]['layer']==(net.num_layers-1)]
        prelayer_nodes = [n for n in net.graph.nodes() if net.graph.node[n]['layer']==(net.num_layers-2)]
        layer_error=net.state[layer_nodes]-target_output
        layer_delta=layer_error*net.nonlinearity_deriv(net.state[layer_nodes])
        net.weights[:,net.output_indices]+=self.alpha*net.state[prelayer_nodes]*layer_delta

        for k in range(net.num_layers-1):
            layer=net.num_layers-k-1
            prelayer_nodes = [n for n in net.graph.nodes() if net.graph.node[n]['layer'] == layer-1]
            layer_nodes = [n for n in net.graph.nodes() if net.graph.node[n]['layer'] == layer]
            layer_indices = [net.node2index(n) for n in layer_nodes]
            layer_error=layer_delta*net.weights[layer_indices,:]
            layer_delta=layer_error*net.nonlinearity_deriv(net.state[layer_nodes])
            net.weights[:,layer_indices]+=self.alpha*net.state[prelayer_nodes]*layer_delta



print(__name__)
if __name__ == '__main__':

    #simple two layer net, with two neurons in input layer, one output neuron in the second layer
    net = Network([2, 1])

    conn = DenseFeedforwardConnector(random=True)
    conn.connect(net)

    #net.step(input=[0.25, -0.9])
    trainer=OnlineBackPropTrainer(0.1)

    in_examples=[[0,0],[0,1],[1,0],[1,1]]
    out_examples=[0,1,1,0]

    for i in range(100):
        ex_idx=np.random.randint(0,high=len(out_examples))
        trainer.train(net, in_examples[ex_idx], out_examples[ex_idx])

