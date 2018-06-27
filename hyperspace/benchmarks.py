import numpy as np


class StyblinksiTang:

    def __init__(self, dims, lower=-5., upper=5.):
        self.dims = dims
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        return f'Styblinkski-Tang function defined over xi ∈ [{self.lower}, {self.upper}] for all i = 1, …, {self.dims}.' 

    def __call__(self, x):
        """
        Styblinski-Tang function.

        Parameters:
        * `x`: [array-like, shape=(,self.dims)]
          Points in domain to be evaluated.

        Notes:
        -----
        Domain:
          Usually evaluated on the hypercube xi ∈ [-5, 5], for all i = 1, …, dims.

        Global minimum:
          f(x*) = -39.16599 * dims  at x* = (-2.903534, ..., -2.903534)

        Reference: 
        ---------
        https://www.sfu.ca/~ssurjano/stybtang.html
        """
        val = 0.0
        for i in range(self.dims):
            val += (np.power(x[i], 4, dtype=np.longdouble) - 16.0 * np.power(x[i], 2, dtype=np.longdouble) + 5.0 * x[i])

        return 0.5 * val
