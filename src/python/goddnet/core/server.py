import numpy
from goddnet.core.model import FeatureModel


class Server(object):

    def __init__(self):

        # numpy random generator
        numpy_rng = numpy.random.RandomState(89677)

        self.feature_model = FeatureModel(numpy_rng)
        self.feature_trainer = Trainer(self.feature_model)

        self.predictor_trainers = dict()
        self.predictor_models = dict()


    def process_train(self, input, model_name=None, label=None):
        # Do unsuperivsed training of all inputs
        self.feature_trainer.train(input)
        if model_name is not None and label is not None:
            if model_name in self.predictor_trainers:
                self.predictor_trainers[model_name].train(self.feature_model.transform(input), label=label)


    def process_predict(self, model_name, input):
        if model_name in self.predictor_models:
            features=self.feature_model.transform(input)
            return self.predictor_models[model_name].predict(features)
        return None

    def add_predictor(self, name, model):
        self.predictor_models[name]=model
        self.predictor_trainers[name]=Trainer(model)


class Trainer(object):

    def __init__(self, model, batch_size=10):
        self.model = model
        self.batch_size = batch_size
        self.queue = list()

    def train(self, input, label=None):

        #add the input to the queue
        self.queue.append( (input, label) )

        #train the model with a mini-batch if the queue is full
        if len(self.queue) >= self.batch_size:
            if self.model.is_unsupervised:
                self.model.train([x[0] for x in self.queue])
            else:
                self.model.train(self.queue)

            #empty the queue
            del self.queue
            self.queue = list()










