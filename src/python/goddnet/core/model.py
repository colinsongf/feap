import theano.tensor as T


class Model(object):
    def __init__(self):
        self.params=[]

    def train(self, data, learning_rate):
        raise NotImplementedError('Use a subclass')

    def predict(self, input):
        raise NotImplementedError('Use a subclass')


class FeatureModel(Model):

    def __init__(self):
        super(FeatureModel,self).__init__()
        self.is_unsupervised=True

    def train(self, data, learning_rate=0.001):
        pass

    def transform(self, data):
        pass


class PredictorModel(Model):

    def __init__(self):
        super(PredictorModel,self).__init__()
        self.y_pred=None
        self.is_unsupervised=False

    def get_updates(self, learning_rate):
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(self.cost(), self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        return updates

    def cost(self):
        pass

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))

    def train(self, data, learning_rate=0.001):
        pass

    def predict(self, data):
        pass



