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
        self.y = T.lvector('y')

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

        learning_rate = T.scalar('learning_rate')  # learning rate to use

        self.train_model = theano.function(inputs=[self.input,self.y,theano.Param(learning_rate, default=0.13)],
            outputs=self.negative_log_likelihood(),
            updates=self.get_updates(learning_rate),
            givens={})
        self.predict = theano.function(inputs=[self.input],
            outputs=self.y_pred,
        )

    def errors(self, y):
        super(LogisticRegression,self).errors(y)

        # check if y is of the correct datatype
        if not y.dtype.startswith('int'):
            raise NotImplementedError()
        return T.mean(T.neq(self.y_pred, y))

    def train(self, data, learning_rate=.13):
        train_set_x = numpy.array([x[0] for x in data])
        train_set_y = numpy.array([x[1] for x in data])
        c = self.train_model(train_set_x,train_set_y,learning_rate=learning_rate)
        return c

    def negative_log_likelihood(self):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(self.y.shape[0]), self.y])
