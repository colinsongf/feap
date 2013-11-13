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
        self.feature_model = None
        self.feature_trainer = None

        self.predictor_trainers = dict()
        self.predictor_models = dict()


    def process_train(self, input, model_name=None, label=None):
        # Do unsuperivsed training of all inputs
        cost_feature=None
        if self.feature_trainer is not None:
            cost_feature=self.feature_trainer.train(input)
        cost_predictor=None
        if model_name is not None and label is not None:
            if model_name in self.predictor_trainers:
                cost_predictor=self.predictor_trainers[model_name].train(self.feature_model.transform(input), label=label)
        return cost_feature,cost_predictor


    def process_predict(self, model_name, input):
        if model_name in self.predictor_models:
            features=self.feature_model.transform(input)
            return self.predictor_models[model_name].predict(features)
        return None

    def set_feature_model(self, model):
        self.feature_model=model
        self.feature_model.is_unsupervised=True
        self.feature_trainer = Trainer(self.feature_model)

    def add_predictor(self, name, model):
        self.predictor_models[name]=model
        self.predictor_trainers[name]=Trainer(model)


class Trainer(object):

    def __init__(self, model, batch_size=600):
        self.model = model
        self.batch_size = batch_size
        self.queue = list()

    def train(self, input, label=None, learning_rate=.1):

        #add the input to the queue
        self.queue.append( (input, label) )

        #train the model with a mini-batch if the queue is full
        if len(self.queue) >= self.batch_size:
            if self.model.is_unsupervised:
                cost=self.model.train([x[0] for x in self.queue], learning_rate=learning_rate)
            else:
                cost=self.model.train(self.queue, learning_rate=learning_rate)

            #empty the queue
            del self.queue
            self.queue = list()

            return cost
        return None

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
    numpy_rng = numpy.random.RandomState(123)
    server=Server()
    server.set_feature_model(SdA(numpy_rng, in_size=784, hidden_sizes=[500,500], out_size=10))
    server.add_predictor('logistic',LogisticRegression(500, 10))
    server.add_predictor('mlp',MLP(numpy_rng, 500, 200, 10))
    server.add_predictor('SdA',SdA(numpy_rng, in_size=500, hidden_sizes=[250, 250], out_size=10))
    print '... pretraining model'
    for i in xrange(pretraining_epochs):
        batch_costs=[]
        for x in train_set_x:
            cost_feature,cost_predictor=server.process_train(x)
            if cost_feature is not None:
                batch_costs.append(cost_feature)
        print('... pretraining epoch %d cost=%.4f' % (i,numpy.mean(batch_costs)))

    print '... training model'
    #server.feature_trainer.batch_size=10
    for i in xrange(training_epochs):
        batch_costs={'logistic':[],'mlp':[],'SdA':[]}
        for x,y in zip(train_set_x,train_set_y):
            cost_feature,cost_predictor=server.process_train(x,model_name='logistic',label=y)
            if cost_predictor is not None:
                batch_costs['logistic'].append(cost_predictor)
            cost_feature,cost_predictor=server.process_train(x,model_name='mlp',label=y)
            if cost_predictor is not None:
                batch_costs['mlp'].append(cost_predictor)
            cost_feature,cost_predictor=server.process_train(x,model_name='SdA',label=y)
            if cost_predictor is not None:
                batch_costs['SdA'].append(cost_predictor)

        print('... training epoch %d: logistic cost=%.4f, mlp cost=%.4f, SdA cost=%.4f' %
              (
                  i,
                  numpy.mean(batch_costs['logistic']),
                  numpy.mean(batch_costs['mlp']),
                  numpy.mean(batch_costs['SdA'])
              )
            )

if __name__ == '__main__':
    test_features(pretraining_epochs=0,training_epochs=20)







