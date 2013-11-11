import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from goddnet.theano.dA import dA
from goddnet.theano.logistic_sgd import LogisticRegression
from goddnet.theano.mlp import HiddenLayer


class Model(object):

    params=[]
    def __init__(self):
        pass

    def train(self, data, learning_rate):
        raise NotImplementedError('Use a subclass')

    def predict(self, input):
        raise NotImplementedError('Use a subclass')


class FeatureModel(Model):

    is_unsupervised=True

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784, hidden_layers_sizes=[500, 500], n_outs=10,
                 corruption_levels=[0.1, 0.2]):

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.corruption_levels=corruption_levels

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input=self.sigmoid_layers[-1].output,
                         n_in=hidden_layers_sizes[-1], n_out=n_outs)

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

        self.pretraining_fns = self.pretraining_functions()

    def pretraining_functions(self):

        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(inputs=[self.x,
                                    theano.Param(corruption_level, default=0.2),
                                    theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def train(self, data, learning_rate=0.001):
        train_set_x = numpy.array(data)
        layer_c=[]
        for i in xrange(self.n_layers):
            layer_c.append(self.pretraining_fns[i](train_set_x,corruption=self.corruption_levels[i],
                               lr=learning_rate))
        return layer_c

    def transform(self, data):
        x=data
        for i in xrange(self.n_layers):
            y = self.dA_layers[i].get_hidden_values(x)
            x=y
        fn = theano.function(inputs=[],
            outputs=x)
        return fn()


class PredictorModel(Model):

    y_pred=None
    is_unsupervised=False

    def get_updates(self, learning_rate):
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(self.cost(), self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        return updates

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))

    def train(self, data, learning_rate=0.001):
        pass

    def predict(self, data):
        pass



