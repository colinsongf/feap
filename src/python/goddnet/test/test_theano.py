import unittest

import numpy as np

import theano as th
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from goddnet.theano.utils import matrix_power


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

        diff = np.abs(ZZ - ZZreal).sum()
        assert diff == 0.0

        #matrix-vector multiplication
        v = T.dvector('v')
        z = T.dot(X, v)
        f = th.function([X, v], z)

        zz = f([[3.3, 4.4], [5.5, 6.6]], [0.5, 0.7])
        assert (zz[0] - 4.73) < 1e-6
        assert (zz[1] - 7.37) < 1e-6

        #matrix square
        A = T.dmatrix('A')
        Z = T.dot(A, A)
        F = th.function([A], Z)
        ZZ = F([[1.0, 2.0], [3.0, 4.0]])
        ZZreal = np.array([[7.0, 10.0], [15.0, 22.0]])
        diff = np.abs(ZZ - ZZreal).sum()
        assert diff < 1e-6

    def testShared(self):

        #create shared memory
        x0 = np.array([10.0, 7.0])
        state = th.shared(x0)

        #create a driven linear dynamical system that passively decays to zero
        A = T.dmatrix('A') #transition matrix
        u = T.dvector('u')
        f = th.function([A, u], state, updates=[(state, T.dot(A, state) + u)])

        AA = np.array([[-0.9, 0.0], [0.0, -0.2]])
        uu = np.array([0.0, 0.0])

        for k in range(1000):
            f(AA, uu)

        final_state = state.get_value()
        assert np.abs(final_state).sum() < 1e-16

    def testRandom(self):

        #create an expression that returns a given gaussian random number with the specified mean and variance
        real_mean = 0.67
        real_std = 0.25
        srng = RandomStreams(seed=1234)
        rv_x = srng.normal((1,))
        f = th.function([], real_mean + real_std*rv_x)

        #take random draws from the distribution
        draws = list()
        for k in range(10000):
            draws.append(f())
        draws = np.array(draws)

        #confirm that the mean and standard deviation are what one would expect
        assert np.abs(draws.mean() - real_mean) < 1e-2
        assert np.abs(draws.std(ddof=1) - real_std) < 1e-2

    def testGrad(self):

        x = T.dvector('x')
        y = T.exp(T.sum(x**2))
        gy = T.grad(y, x)

        f = th.function([x], gy)

        f_real = th.function([x], 2*x*T.exp(T.sum(x**2)))

        f1 = f([0.5, 0.2])
        f1real = f_real([0.5, 0.2])
        diff = np.abs(f1 - f1real).sum()

        assert diff == 0.0

    def testScan(self):

        A = [[1, 2], [3, 4]]
        A4 = matrix_power(A, 4)
        Anp = np.linalg.matrix_power(A, 4)
        diff = np.abs(A4 - Anp).sum()
        assert diff < 1e-6


if __name__ == '__main__':
    unittest.main()