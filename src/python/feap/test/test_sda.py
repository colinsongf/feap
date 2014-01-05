import matplotlib.pyplot as plt
import numpy as np
import unittest
from feap.core.server import Trainer
from feap.models.SdA import SdA

class TestSdA(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sda(self):
        np.random.seed(12345)
        numpy_rng = np.random.RandomState(123)
        #generate some fake data
        nsamps = 30000
        n_in = 20
        n_hidden =[50,50]
        n_out = 3
        W = np.random.randn(n_out, n_in)
        X = np.random.randn(n_in, nsamps)
        noise = np.random.randn(n_out, nsamps)*1
        e = np.exp(np.array(np.dot(W, X) + noise))
        Y = e / np.sum(e)
        Y=np.argmax(Y,axis=0)

        #construct a model and trainer
        model = SdA(numpy_rng, in_size=n_in, hidden_sizes=n_hidden, out_size=n_out, unsupervised_epochs=10,
            unsupervised_learning_rate=.001)
        trainer = Trainer(model, batch_size=100)

        #train the model in mini batches
        nvalid = 500

        pretrain_errors=list()
        train_errors = list()
        valid_errors = list()
        model.is_unsupervised=True
        for k in range(nsamps-nvalid):
            print 'Training sample %d' % (k+1)
            c = trainer.train(X[:, k], Y[k], learning_rate=0.001)
            if c is not None:
                pretrain_errors.append(c)
        plt.figure()
        plt.plot(pretrain_errors, 'k-', linewidth=2.0)
        plt.title('Errors During Pretraining')
        plt.xlabel('Epoch (%d iterations)' % trainer.batch_size)
        plt.axis('tight')
        plt.show()

        model.is_unsupervised=False
        #trainer.training_epochs=1
        for k in range(nsamps-nvalid):
            print 'Training sample %d' % (k+1)
            c = trainer.train(X[:, k], Y[k], learning_rate=0.3)
            if c is not None:
                train_errors.append(c)
                #compute error on validation set
                Ypred = list()
                for m in range(nvalid):
                    yhat = model.predict(X[:, -(nvalid-m)])
                    Ypred.append(yhat)
                Ypred = np.array(Ypred)
                msqerr = ((Y[-nvalid:] - Ypred.T)**2).mean(axis=1).mean()
                valid_errors.append(msqerr)

        plt.figure()
        plt.plot(train_errors, 'k-', linewidth=2.0)
        plt.plot(valid_errors, 'r-')
        plt.legend(['Training', 'Validation'])
        plt.title('Errors During Training')
        plt.xlabel('Epoch (%d iterations)' % trainer.batch_size)
        plt.axis('tight')
        plt.show()
