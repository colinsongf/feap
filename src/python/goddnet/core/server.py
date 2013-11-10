import gzip
import numpy
import os
import cPickle
from goddnet.core.model import FeatureModel
from goddnet.models.logistic import LogisticRegression, LogisticNegativeLogLikelihood


class Server(object):

    def __init__(self):

        # numpy random generator
        numpy_rng = numpy.random.RandomState(89677)
        print '... building the model'
        self.feature_model = FeatureModel(numpy_rng)
        self.feature_trainer = Trainer(self.feature_model)

        self.predictor_trainers = dict()
        self.predictor_models = dict()


    def process_train(self, input, model_name=None, label=None):
        # Do unsuperivsed training of all inputs
        c_u=self.feature_trainer.train(input)
        c_s=0
        if model_name is not None and label is not None:
            if model_name in self.predictor_trainers:
                c_s=self.predictor_trainers[model_name].train(self.feature_model.transform(input), label=label)
        return c_u,c_s


    def process_predict(self, model_name, input):
        if model_name in self.predictor_models:
            features=self.feature_model.transform(input)
            return self.predictor_models[model_name].predict(features)
        return None

    def add_predictor(self, name, model):
        self.predictor_models[name]=model
        self.predictor_trainers[name]=Trainer(model, batch_size=10)


class Trainer(object):

    def __init__(self, model, batch_size=600):
        self.model = model
        self.batch_size = batch_size
        self.queue = list()

    def train(self, input, label=None):

        #add the input to the queue
        self.queue.append( (input, label) )

        c=[]
        #train the model with a mini-batch if the queue is full
        if len(self.queue) >= self.batch_size:
            if self.model.is_unsupervised:
                c.extend(self.model.train([x[0] for x in self.queue]))
            else:
                self.model.train(self.queue)

            #empty the queue
            del self.queue
            self.queue = list()
        return c

def test_features(pretraining_epochs=15, training_epochs=15, dataset='../../data/mnist.pkl.gz'):
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_set_x, train_set_y = train_set
    server=Server()
    server.add_predictor('logistic',LogisticNegativeLogLikelihood(500, 10))
    print '... pretraining model'
    for i in xrange(pretraining_epochs):
        c=[]
        for x in train_set_x:
            c_u,c_s=server.process_train(x)
            c.extend(c_u)
        print('... pretraining epoch %d cost=%.4f' % (i,numpy.mean(c)))

    print '... training model'
    server.feature_trainer.batch_size=10
    for i in xrange(training_epochs):
        c=[]
        for x,y in zip(train_set_x,train_set_y):
            c_u,c_s=server.process_train(x,model_name='logistic',label=y)
            c.extend(c_s)
        print('... training epoch %d cost=%.4f' % (i,numpy.mean(c)))

if __name__ == '__main__':
    test_features(pretraining_epochs=5,training_epochs=5)






