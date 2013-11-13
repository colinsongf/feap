import unittest

import numpy as np

import matplotlib.pyplot as plt

from feap.core.server import Trainer
from feap.models.regression import LinearRegression


class TestRegression(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_linear(self):

        np.random.seed(12345)

        #generate some fake data
        nsamps = 10000
        n_in = 20
        n_out = 10
        W = np.random.randn(n_out, n_in)
        X = np.random.randn(n_in, nsamps)
        noise = np.random.randn(n_out, nsamps)*1e-1
        Y = np.dot(W, X) + noise

        #construct a model and trainer
        model = LinearRegression(n_in, n_out)
        trainer = Trainer(model, batch_size=100)

        #train the model in mini batches
        nvalid = 500

        train_errors = list()
        valid_errors = list()
        for k in range(nsamps-nvalid):
            print 'Training sample %d' % (k+1)
            c = trainer.train(X[:, k], Y[:, k])
            if k % trainer.batch_size == 0:
                if c is not None:
                    train_errors.append(c)
                #compute error on validation set
                Ypred = list()
                for m in range(nvalid):
                    yhat = model.predict(X[:, -(nvalid-m)])
                    Ypred.append(yhat)
                Ypred = np.array(Ypred)
                msqerr = ((Y[:, -nvalid:] - Ypred.T)**2).mean(axis=1).mean()
                valid_errors.append(msqerr)

        plt.figure()
        plt.plot(train_errors, 'k-', linewidth=2.0)
        plt.plot(valid_errors, 'r-')
        plt.legend(['Training', 'Validation'])
        plt.title('Errors During Training')
        plt.xlabel('Epoch (%d iterations)' % trainer.batch_size)
        plt.axis('tight')
        plt.show()