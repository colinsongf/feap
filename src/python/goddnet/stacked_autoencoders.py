import numpy as np
from goddnet.bp import FeedforwardNetwork, BackPropTrainer, half_sq_error, half_sq_error_deriv

class StackedAutoencoderNetwork():
    """
    Represents a network of stacked autoencoders
    """
    def __init__(self, autoencoder_shapes):
        self.n_autoencoders=len(autoencoder_shapes)
        # Init autoencoders
        self.autoencoders=[]
        for i in range(len(autoencoder_shapes)):
            self.autoencoders.append(FeedforwardNetwork(autoencoder_shapes[i]))
        # Init weights between autoencoders
        self.weights=[]
        self.c=[]
        for i in range(len(autoencoder_shapes)-1):
            self.weights.append(.05*np.random.randn(autoencoder_shapes[i][1],autoencoder_shapes[i+1][0]-1))
            self.c.append(np.zeros([autoencoder_shapes[i][1],autoencoder_shapes[i+1][0]-1]))

    def run (self, inputs):
        """
        Run network on given inputs
        """
        if len(inputs) != self.autoencoders[0].shape[0]-1:
            print 'incorrect number of inputs'
        layer_input=inputs
        # Run each autoencoder
        for i in range(self.n_autoencoders):
            self.autoencoders[i].run(layer_input)
            # Use hidden layer of this autoencoder as input to next one
            layer_input=self.autoencoders[i].node_activations[1]

        # Return output of final autoencoder
        return self.autoencoders[-1].node_activations[-1]

    def print_weights(self):
        """
        Print all weights between autoencoders
        """
        for i in range(self.n_autoencoders-1):
            print'Autoencoder %d -> Autoencoder %d weights' % (i,i+1)
            for j in range(self.autoencoders[i].shape[1]):
                print self.weights[i][j,:]
            print ''

class StackedAutoencoderTrainer(BackPropTrainer):
    """
    Trains a stacked autoencoder using backprop
    """
    def __init__(self, learning_rate=0.1, momentum=0.05, error=half_sq_error, error_deriv=half_sq_error_deriv):
        BackPropTrainer.__init__(self, learning_rate=learning_rate, momentum=momentum, error=error,
            error_deriv=error_deriv)

    def init_train_simultaneous(self, net, patterns, max_iterations = 10000):
        """
        Initial training of autoencoders simultaneously on given patterns
        """
        for i in range(max_iterations):
            total_error=0
            for (in_pattern,out_pattern) in patterns:
                pattern_err=0.0
                for j in range(net.n_autoencoders-1):
                    # Train autoencoder on pattern
                    trainer=BackPropTrainer(learning_rate=0.1, momentum=0.0)
                    # Run autoencoder with pattern
                    net.autoencoders[j].run(in_pattern)
                    # Train
                    (err,layer_deltas) = trainer.backPropagate(net.autoencoders[j], out_pattern)
                    pattern_err+=err/(net.n_autoencoders-1.0)
                    # Run again to get activity with new weights
                    net.autoencoders[j].run(in_pattern)
                    # Use hidden layer activity and corrupted copy as training for next autoencoder
                    in_pattern=net.autoencoders[j].node_activations[1]
                    out_pattern=in_pattern+.1*np.random.randn(len(in_pattern))
                total_error+=pattern_err
            if i % 50 == 0:
                print 'Combined error', total_error
            if i>0 and total_error<.01:
                break

    def init_train_sequence(self, net, patterns, max_iterations=1000):
        """
        Initial training of autoencoders in sequence
        """
        train_patterns=patterns
        for j in range(net.n_autoencoders-1):
            print('Training Autoencoder %d' % j)
            # Train autoencoder
            trainer=BackPropTrainer(learning_rate=0.025, momentum=0.0)
            trainer.train(net.autoencoders[j], train_patterns, max_iterations=max_iterations, err_thresh=0.01)
            train_patterns=[]
            # Create patterns to train next autoencoder
            for i in range(len(patterns)):
                in_pattern=patterns[i][0]
                net.run(in_pattern)
                new_in_pattern=net.autoencoders[j].node_activations[1]
                new_out_pattern=new_in_pattern+.1*np.random.randn(len(new_in_pattern))
                train_patterns.append([new_in_pattern,new_out_pattern])

    def backPropagate(self, net, targets):
        """
        Trains weights between autoencoders
        """
        # Train weights in last autoencoder
        trainer=BackPropTrainer(learning_rate=self.learning_rate, momentum=self.momentum, error=self.error, error_deriv=self.error_deriv)
        (error,layer_deltas)=trainer.backPropagate(net.autoencoders[-1],targets)

        # Train weights from hidden layer of second to last autoencoder to input layer of last autoencoder
        postlayer_weights=net.autoencoders[-1].weights[0][:-1,:] # Exclude bias
        inputlayer=net.autoencoders[-2].node_activations[1]
        hiddenlayer=net.autoencoders[-1].node_activations[0][:-1] # Exclude bias
        (layer_deltas,net.weights[-1],net.c[-1])=self.adjust_weights_to_hiddenlayer(layer_deltas,
            postlayer_weights,inputlayer[:,np.newaxis],hiddenlayer,net.weights[-1],net.c[-1],
            net.autoencoders[-1].nonlinearity_deriv)

        if net.n_autoencoders>2:
            for weight_idx in range(net.n_autoencoders-3,-1,-1):
                encoder_idx=weight_idx+1

                # Train weights from hidden layer of previous autoencoder to input layer of this autoencoder
                postlayer_weights=net.autoencoders[encoder_idx].weights[0][:-1,:] # Exclude bias
                inputlayer=net.autoencoders[weight_idx].node_activations[1]
                hiddenlayer=net.autoencoders[encoder_idx].node_activations[0][:-1] # Exclude bias
                (layer_deltas,net.weights[weight_idx], net.c[weight_idx])=self.adjust_weights_to_hiddenlayer(layer_deltas,
                    postlayer_weights,inputlayer[:,np.newaxis], hiddenlayer,net.weights[weight_idx], net.c[weight_idx],
                    net.autoencoders[encoder_idx].nonlinearity_deriv)

        # calc combined error
        total_error=np.sum(self.error(targets,net.autoencoders[-1].node_activations[-1]))
        return total_error,layer_deltas


def test_run():
    net=StackedAutoencoderNetwork([[5,10,5],[10,8,10]])
    print(net.run([0,1,0,1,1]))
    net.print_weights()

def test_init_train():
    input_patterns=[[0,0],[0,1],[1,0],[1,1]]
    patterns=[]
    for i in range(len(input_patterns)):
        for j in range(50):
            test_input=input_patterns[i]+.1*np.random.randn(len(input_patterns[i]))
            test_output=input_patterns[i]
            patterns.append([test_input,test_output])
    net=StackedAutoencoderNetwork([[2,10,2],[10,20,10],[20,40,20]])
    trainer = StackedAutoencoderTrainer(learning_rate=0.05, momentum=0.0)
    trainer.init_train_simultaneous(net,patterns)
    #trainer.init_train_sequence(net,patterns)

def test_train():
    final_patterns=[[[0,0],[0]],
                    [[0,1],[1]],
                    [[1,0],[1]],
                    [[1,1],[0]]]
    init_patterns=[]
    for i in range(len(final_patterns)):
        in_pattern=final_patterns[i][0]
        for j in range(20):
            out_pattern=in_pattern+.1*np.random.randn(len(in_pattern))
            init_patterns.append([in_pattern,out_pattern])
    net=StackedAutoencoderNetwork([[2,10,2],[10,5,10],[5,10,1]])
    trainer=StackedAutoencoderTrainer(learning_rate=0.1, momentum=0.0)
    trainer.init_train_sequence(net, init_patterns)
    trainer.train(net, final_patterns)

if __name__ == "__main__":
    #test_init_train()
    test_train()
