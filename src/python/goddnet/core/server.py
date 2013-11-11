import gzip
import numpy
import os
import cPickle
from goddnet.models.SdA import SdA
from goddnet.models.mlp import MLP
from goddnet.models.regression import LogisticRegression


class Server(object):

    def __init__(self):

        print '... building the model'
        numpy_rng = numpy.random.RandomState(123)
        self.feature_model = SdA(numpy_rng, in_size=784, hidden_sizes=[500,500], out_size=10)
        self.feature_model.is_unsupervised=True
        self.feature_trainer = Trainer(self.feature_model)

        self.predictor_trainers = dict()
        self.predictor_models = dict()


    def process_train(self, input, model_name=None, label=None):
        # Do unsuperivsed training of all inputs
        c_u=self.feature_trainer.train(input)
        c_s=[]
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
        self.predictor_trainers[name]=Trainer(model)


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
                c.append(self.model.train([x[0] for x in self.queue]))
            else:
                c.append(self.model.train(self.queue))

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
    server.add_predictor('logistic',LogisticRegression(500, 10))
    numpy_rng = numpy.random.RandomState(123)
    server.add_predictor('mlp',MLP(numpy_rng, 500, 200, 10))
    server.add_predictor('SdA',SdA(numpy_rng, in_size=500, hidden_sizes=[250, 250], out_size=10))
    print '... pretraining model'
    for i in xrange(pretraining_epochs):
        c=[]
        for x in train_set_x:
            c_u,c_s=server.process_train(x)
            c.extend(c_u)
        print('... pretraining epoch %d cost=%.4f' % (i,numpy.mean(c)))

    print '... training model'
    #server.feature_trainer.batch_size=10
    for i in xrange(training_epochs):
        c_log=[]
        c_mlp=[]
        c_sda=[]
        for x,y in zip(train_set_x,train_set_y):
            c_u,c_s=server.process_train(x,model_name='logistic',label=y)
            c_log.extend(c_s)
            c_u,c_s=server.process_train(x,model_name='mlp',label=y)
            c_mlp.extend(c_s)
            c_u,c_s=server.process_train(x,model_name='SdA',label=y)
            c_sda.extend(c_s)

        print('... training epoch %d: logistic cost=%.4f, mlp cost=%.4f, SdA cost=%.4f' % (i,numpy.mean(c_log),numpy.mean(c_mlp),numpy.mean(c_sda)))

if __name__ == '__main__':
    test_features(pretraining_epochs=15,training_epochs=15)







