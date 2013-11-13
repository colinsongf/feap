import numpy as np

import theano as th
import theano.tensor as T

def matrix_power_function():
    """
        Returns a Theano function that computes matrix_power(A, k) = A**k, where A is a tensor and k is the
        exponent. Both A and k must be symbolic.
    """

    def matrix_power_inner_func(Aprod, A):
        return T.dot(Aprod, A)

    A = T.dmatrix('A')
    k = T.iscalar('k')

    result, updates = th.scan(fn=matrix_power_inner_func, outputs_info=T.identity_like(A), non_sequences=A, n_steps=k)
    final_result = result[-1]

    power = th.function(inputs=[A, k], outputs=final_result, updates=updates, mode='DebugMode')

    return power

matrix_power = matrix_power_function()