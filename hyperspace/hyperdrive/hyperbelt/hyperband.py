import numpy as np
from math import log, ceil

from skopt.space import Space
from hyperspace.kepler import create_result


def hyperband(objective, space, n_evaluations, max_iter=100, eta=3,
              random_state=0, verbose=True, debug=False, rank=None):
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
    # Convert space into search dimensinons
    space = Space(space)
    #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
    yi = []
    Xi = []
    for s in reversed(range(s_max+1)):
        n = int(ceil(int(B/max_iter/(s+1))*eta**s)) # initial number of configurations
        r = max_iter*eta**(-s) # initial number of iterations to run configurations for
        #
        T = space.rvs(n, random_state)
        # Begin Finite Horizon Successive Halving with (n,r)
        for i in range(s+1):
            # Run each of the n_i configs for r_i iterations and keep best n_i/eta
            n_i = n*eta**(-i)
            r_i = r*eta**(i)

            iter_result = [objective(t, r_i) for t in T]
            yi.append(iter_result)
            Xi.append(T)

            # Get next hyperparameter configurations
            T = [ T[i] for i in np.argsort(iter_result)[0:int( n_i/eta )] ]
            print(f'num hyperparameter configs: {len(T)}')

            if verbose:
                print(f'Rank {rank} Iteration number: {i}, Func value: {min(iter_result)}, num configs: {len(iter_result)}\n')
            if debug:
                print(f'Number of hyperparameter configurations: {len(T)}')

        result = create_result(Xi, yi, n_evaluations=n_evaluations, space=space, rng=random_state)
        # End Finite Horizon Successive Halving with (n,r)
        return result

