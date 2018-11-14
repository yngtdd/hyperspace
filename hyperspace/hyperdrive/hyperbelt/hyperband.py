import numpy as np
from math import log, ceil

from skopt.space import Space
from hyperspace.kepler import create_result


def hyperband(objective, space, max_iter=100, eta=3, random_state=0,
              verbose=True, n_evaluations=None, rank=0):
    """
    Hyperband algorithm as defined by Kevin Jamieson.

    Parameters:
    ----------
    * `objective`: [function]
        Objective function to be minimized.

    * `space`: [hyperspace.space]
        Hyperparameter search space bounds.

    * `max_iter`: [int]
        Maximum number of iterations.

    * `eta`: [int]

    Returns:
    -------
    * `result` [`OptimizeResult`, scipy object]

    Reference:
    ---------
    http://people.eecs.berkeley.edu/~kjamieson/hyperband.html
    """
    logeta = lambda x: log(x)/log(eta)
    s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
    B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)
    # If space is the original list of tuples, convert to Space()
    if isinstance(space, list):
        space = Space(space)
    #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
    yi = []
    Xi = []
    for s in reversed(range(s_max+1)):
        n = int(ceil(int(B/max_iter/(s+1))*eta**s)) # initial number of configurations
        r = max_iter*eta**(-s) # initial number of iterations to run configurations for
        # Begin Finite Horizon Successive Halving with (n,r)
        for i in range(s+1):
            # Run each of the n_i configs for r_i iterations and keep best n_i/eta
            n_i = ceil(n*eta**(-i))
            r_i = ceil(r*eta**(i))

            T = space.rvs(ceil(n_i), random_state)
            iter_result = [objective(t, r_i) for t in T]
            yi.append(iter_result)
            Xi.append(T)

            # Get next hyperparameter configurations
            T = [ T[i] for i in np.argsort(iter_result)[0:int( n_i/eta )] ]

            if verbose and rank == 0:
                print(f'Iteration number: {i}, Epochs per config: {r_i}, Num configs: {n_i}, Incumbent: {min(iter_result)}')

        result = create_result(Xi, yi, n_evaluations=n_evaluations, space=space, rng=random_state)
        # End Finite Horizon Successive Halving with (n,r)
        return result
