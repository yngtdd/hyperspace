from skopt import gp_minimize
from skopt import gbrt_minimize
from skopt import forest_minimize
from skopt import dummy_minimize


def minimize(objective, space, model="GP", n_calls=50, verbose=False,
             deadline=None, x_init=None, n_random_starts=None,
             sampler=None, n_samples=None, hyperbounds=None, name=None, random_state=0):
    """
    Surrogate models for objective functions.
    """
    # Thanks Guido for refusing to believe in switch statements.
    # Case 0
    if model == "GP":
        result = gp_minimize(objective, space, n_calls=n_calls, verbose=verbose,
                             callback=deadline, x0=x_init, n_random_starts=n_random_starts,
                             random_state=random_state)
    # Case 1
    elif model == "RF":
        result = forest_minimize(objective, space, n_calls=n_calls, verbose=verbose,
                                 callback=deadline, x0=x_init, n_random_starts=n_random_starts,
                                 random_state=random_state)
    # Case 2
    elif model == "GRBRT":
        result = gbrt_minimize(objective, space, n_calls=n_calls, verbose=verbose,
                               callback=deadline, x0=x_init, n_random_starts=n_random_starts,
                               random_state=random_state)
    # Case 3
    elif model == "RAND":
        result = dummy_minimize(objective, space, n_calls=n_calls, verbose=verbose,
                                callback=deadline, x0=x_init, random_state=random_state)
    else:
        raise ValueError("Invalid model {}. Read the documentation for "
                         "supported models.".format(model))

    return result
