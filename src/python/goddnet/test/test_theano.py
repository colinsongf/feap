import unittest

import numpy as np
import theano as th
import theano.tensor as T


class SDAE(object):

    def __init__(self, nIn):
        pass

    def train(self, samples):
        pass


class TestTheano(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testSimple(self):

        #scalar addition
        x = T.dscalar('x')
        y = T.dscalar('y')
        z = x + y
        f = th.function([x, y], z)

        zz = f(2, 3)
        assert zz == 5.0

        #matrix multiplication
        X = T.dmatrix('X')
        Y = T.dmatrix('Y')

        Z = T.dot(X, Y)
        F = th.function([X, Y], Z)

        ZZ = F([[3.3, 4.4], [5.5, 6.6]], [[2.0, 0.0], [0.0, 1.0]])
        ZZreal = np.array([[6.6, 4.4], [11.0, 6.6]])

        diff = (ZZ - ZZreal).sum()
        assert diff == 0.0

        #matrix-vector multiplication
        v = T.dvector('v')
        z = T.dot(X, v)
        f = th.function([X, v], z)

        zz = f([[3.3, 4.4], [5.5, 6.6]], [0.5, 0.7])
        assert (zz[0] - 4.73) < 1e-6
        assert (zz[1] - 7.37) < 1e-6

        #matrix power
        Z = X**1000
        F = th.function([X], Z)
        ZZ = F([[0.4, 0.6], [0.33, 0.67]])
        ZZreal = np.array([[ 0.35483871, 0.64516129], [ 0.35483871,  0.64516129]])
        diff = (ZZ - ZZreal).sum()
        assert diff < 1e-6


if __name__ == '__main__':
    unittest.main()