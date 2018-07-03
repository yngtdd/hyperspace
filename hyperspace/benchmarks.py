import math
import numpy as np


class StyblinskiTang:
    """
    Styblinski-Tang function.

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
    def __init__(self, dims, lower=-5., upper=5.):
        self.dims = dims
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        return f'Styblinski-Tang function defined over xi ∈ [{self.lower}, {self.upper}] for all i = 1, …, {self.dims}.' 

    def __call__(self, x):
        """
        Query the Styblinski-Tang function at x.
        
        Parameters:
        ----------
        * `x`: [array-like, shape=(,self.dims)]
          Points in domain to be evaluated.
        """
        val = 0.0
        for i in range(self.dims):
            val += (np.power(x[i], 4, dtype=np.longdouble) - 16.0 * np.power(x[i], 2, dtype=np.longdouble) + 5.0 * x[i])

        return 0.5 * val


class Sphere:
    """
    Sphere function.

    Notes:
    -----
    The Sphere function has `dims` local minima except for the global one. 
    It is continuous, convex and unimodal.

    Domain:
      Usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, dims.

    Global minimum:
      f(x*) = 0  at x* = (0, …, 0)

    Reference:
    ---------
    https://www.sfu.ca/~ssurjano/spheref.html
    """
    def __init__(self, dims, lower=-5.12, upper=5.12):
        self.dims = dims
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        return f'Sphere function defined over xi ∈ [{self.lower}, {self.upper}] for all i = 1, …, {self.dims}.'

    def __call__(self, x):
        """
        Query the Sphere function at x.

        Parameters:
        ----------
        * `x`: [array-like, shape=(,self.dims)]
          Points in domain to be evaluated.
        """
        val = 0.0
        for i in range(self.dims):
            val += x[i]**2

        return val


class Rosenbrock:
    """
    Rosenbrock function.

    Notes:
    -----
    The function is unimodal, and the global minimum lies in a narrow, parabolic valley. 
    However, even though this valley is easy to find, convergence to the minimum is difficult (Picheny et al., 2012).

    Domain:
      Usually evaluated on the hypercube xi ∈ [-5, 10], for all i = 1, …, dims.
      However, it is sometimes evaluated on the hypercube xi ∈ [-2.048, 2.048], fro all i = 1, …, dims.

    Global minimum:
      f(x*) = 0  at x* = (1, …, 1)

    Reference:
    ---------
    https://www.sfu.ca/~ssurjano/rosen.html
    """
    def __init__(self, dims, lower=-5., upper=10.):
        self.dims = dims
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        return f'Rosenbrock function defined over xi ∈ [{self.lower}, {self.upper}] for all i = 1, …, {self.dims}.'

    def __call__(self, x):
        """
        Query the Rosenbrock function at x.

        Parameters:
        ----------
        * `x`: [array-like, shape=(,self.dims)]
          Points in domain to be evaluated.
        """
        val = 0.0
        for i in range(self.dims-1):
            val += (100 * np.power(x[i+1] - x[i]**2, 2, dtype=np.longdouble)) - np.power(1 - x[i], 2, dtype=np.longdouble) 
        
        return val


class Rastigrin:
    """
    Rastigrin function.

    Notes:
    -----
    The Rastrigin function has several local minima. It is highly multimodal, 
    but locations of the minima are regularly distributed.

    Domain:
      Usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, dims.

    Global minimum:
      f(x*) = 0  at x* = (0, …, 0)

    Reference:
    ---------
    https://www.sfu.ca/~ssurjano/rastr.html
    """
    def __init__(self, dims, lower=-5.12, upper=5.12):
        self.dims = dims
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        return f'Rastigrin function defined over xi ∈ [{self.lower}, {self.upper}] for all i = 1, …, {self.dims}.'

    def __call__(self, x):
        """
        Query the Rosenbrock function at x.

        Parameters:
        ----------
        * `x`: [array-like, shape=(,self.dims)]
          Points in domain to be evaluated.
        """
        val = 0.0
        for i in range(self.dims):
            val += x[i]**2 - 10 * np.cos(2 * math.pi * x[i])

        return 10 * self.dims + val
