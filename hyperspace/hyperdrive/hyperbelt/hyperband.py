import numpy as np
from math import log, ceil

from hyperspace.hyperdrive.models.minimize import minimize


def hyperband(objective, space, model="GP", max_iter=50, eta=3,
              x_init=None, n_random_starts=None, verbose=False, random_state=0, debug=False):
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

    #### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
    for s in reversed(range(s_max+1)):
        n = int(ceil(int(B/max_iter/(s+1))*eta**s)) # initial number of configurations
        r = max_iter*eta**(-s) # initial number of iterations to run configurations for

        # Begin Finite Horizon Successive Halving with (n,r)
        for i in range(s+1):
            # Run each of the n_i configs for r_i iterations and keep best n_i/eta
            n_i = n*eta**(-i)
            r_i = r*eta**(i)

            all_results = []
            if i == 0:
                # Let Scikit-Optimize generate the initial hyperparameter set.
                result = minimize(objective, space, model=model, n_calls=n,
                                  x_init=x_init, n_random_starts=n_random_starts,
                                  random_state=random_state, verbose=verbose)

                all_results.append(result)
                # Get hyperparameters used in random search
                T = result.x_iters
            else:
                # use the random params from the hyperband algo.
                result = minimize(objective, space, model=model,
                                  n_calls=int( n_i/eta ), x_init=T, verbose=verbose)

                all_results.append(result)

            # Get next hyperparameter configurations
            T = [ T[i] for i in np.argsort(result.func_vals)[0:int( n_i/eta )] ]

            if verbose:
                print('Iteration number: {}, Func value: {}'.format(i, result.fun))
            if debug:
                print('Number of hyperparameter configurations: {}'.format(len(T)))

        # End Finite Horizon Successive Halving with (n,r)
        return result
