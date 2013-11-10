import numpy
import theano
import theano.tensor as T
from goddnet.core.model import PredictorModel

class LogisticRegression(PredictorModel):
    """
    Multiclass logistic regression
    """
    def __init__(self, in_size, out_size):
        """
        input = input examples (symbol)
        in_size = size of input
        out_size = size of output
        """
        super(LogisticRegression,self).__init__()

        self.input=T.matrix('x')
        self.y = T.ivector('y')

        # Init weights
        self.W = theano.shared(value=numpy.zeros((in_size, out_size), dtype=theano.config.floatX),
            name='W', borrow=True)

        # Init bias
        self.b = theano.shared(value=numpy.zeros((out_size,), dtype=theano.config.floatX),
            name='b', borrow=True)

        # Class probabilities - softmax of input*weights+bias
        self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)

        # Class prediction - maximum probability
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # model parameters
        self.params.extend([self.W, self.b])

    def errors(self, y):
        super(LogisticRegression,self).errors(y)

        # check if y is of the correct datatype
        if not y.dtype.startswith('int'):
            raise NotImplementedError()

    def train(self, data, learning_rate=0.13):
        train_set_x = theano.shared(numpy.asarray([x[0] for x in data], dtype=theano.config.floatX), borrow=True)
        train_set_y = theano.shared(numpy.asarray([x[1] for x in data], dtype=theano.config.floatX), borrow=True)
        train_set_y = T.cast(train_set_y, 'int32')

        train_model = theano.function(inputs=[],
            outputs=self.cost(),
            updates=self.get_updates(learning_rate),
            givens={self.input:train_set_x,
                    self.y:train_set_y})
        c = train_model()
        return c

    def predict(self, data):
        fn = theano.function(inputs=[],
            outputs=T.argmax(T.nnet.softmax(T.dot(data, self.W) + self.b),axis=1),
            )
        return fn()


class LogisticNegativeLogLikelihood(LogisticRegression):
    def cost(self):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(self.y.shape[0]), self.y])

    def errors(self, y):
        super(LogisticNegativeLogLikelihood,self).errors(y)
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return T.mean(T.neq(self.y_pred, y))