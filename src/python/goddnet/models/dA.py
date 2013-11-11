import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from goddnet.core.model import Model

class dA(Model):
    """
    Denoising autoencoder
    """
    def __init__(self, in_size, hidden_size, numpy_rng, input=None, W=None, bhid=None, bvis=None):
        super(Model,self).__init__()
        self.is_unsupervised=True
        self.in_size = in_size
        self.n_hidden = hidden_size

        # create a Theano random generator that gives symbolic random values
        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (hidden_size + in_size)),
                high=4 * numpy.sqrt(6. / (hidden_size + in_size)),
                size=(in_size, hidden_size)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(in_size,
                dtype=theano.config.floatX),
                borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(hidden_size,
                dtype=theano.config.floatX),
                name='b',
                borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.input = T.dmatrix(name='input')
        else:
            self.input = input

        self.params.extend([self.W, self.b, self.b_prime])

        self.hidden=self.get_hidden_values(self.input)
        self.reconstructed=self.get_reconstructed_input(self.hidden)

        learning_rate = T.scalar('learning_rate')  # learning rate to use
        corruption_level = T.scalar('corruption_level')  # learning rate to use

        self.train_model = theano.function(inputs=[self.input,theano.Param(learning_rate, default=0.13),theano.Param(corruption_level, default=0.1)],
            outputs=self.cost(corruption_level),
            updates=self.get_updates(learning_rate, corruption_level),
            givens={})

        self.transform = theano.function(inputs=[self.input],
            outputs=self.reconstructed,
        )

    def get_corrupted_input(self, input, corruption_level):
        return  self.theano_rng.binomial(size=input.shape, n=1,
            p=1 - corruption_level,
            dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_updates(self, learning_rate, corruption_level):
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(self.cost(corruption_level), self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        return updates

    def cost(self, corruption_level):
        tilde_x = self.get_corrupted_input(self.input, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.input * T.log(z) + (1 - self.input) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        return T.mean(L)

    def train(self, data, learning_rate=.13, corruption_level=.1):
        c = self.train_model(numpy.array(data),learning_rate=learning_rate, corruption_level=corruption_level)
        return c


