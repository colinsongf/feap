import numpy as np

import theano
import theano.tensor as T

def matrix_power(A, k):
    """
        Returns a symbolic function that computes A**k, where A is a tensor and n is the exponent. Both A and k must
        be symbolic. TODO: make this work
    """

    pfunc = lambda X: np.dot(X, X)
    result, updates = theano.scan(fn=pfunc, outputs_info=T.ones_like(A), non_sequences=A, n_steps=k)
    final_result = result[-1]
    power = theano.function(inputs=[A,k], outputs=final_result, updates=updates)

    return power
