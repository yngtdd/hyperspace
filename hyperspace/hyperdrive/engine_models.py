from skopt import gp_minimize
from skopt import gbrt_minimize
from skopt import forest_minimize
from skopt import dummy_minimize
from skopt import dump
from skopt.callbacks import DeadlineStopper

from hyperspace.rover.latin_hypercube_sampler import lhs_start


def minimize(objective, space, rank, results_path, model="GP", n_iterations=50,
             verbose=False, deadline=None, sampler=None, n_samples=None,
             hyperbounds=None, name=None, random_state=0):

    if deadline:
        deadline = DeadlineStopper(deadline)

    if sampler:
        # Get initial points in the obj. function domain via latin hypercube sampling
        init_points = lhs_start(hyperbounds, n_samples)
        n_rand = 10 - len(init_points)
    else:
        init_points = None
        n_rand = 10

    # Thanks Guido for refusing to believe in switch statements.
    # Case 0
    if model == "GP":
        result = gp_minimize(objective, space, n_calls=n_iterations, verbose=verbose,
                             callback=deadline, x0=init_points, n_random_starts=n_rand,
                             random_state=random_state)
    # Case 1
    elif model == "RF":
        result = forest_minimize(objective, space, n_calls=n_iterations, verbose=verbose,
                                 callback=deadline, x0=init_points, n_random_starts=n_rand,
                                 random_state=random_state)
    # Case 2
    elif model == "GRBRT":
        result = gbrt_minimize(objective, space, n_calls=n_iterations, verbose=verbose,
                               callback=deadline, x0=init_points, n_random_starts=n_rand,
                               random_state=random_state)
    # Case 3
    elif model == "RAND":
        result = dummy_minimize(objective, space, n_calls=n_iterations, verbose=verbose,
                                callback=deadline, x0=init_points, n_random_starts=n_rand,
                                random_state=random_state)
    else:
        raise ValueError("Invalid model {}. Read the documentation for "
                         "supported models.".format(model))

    # Each worker will independently write their results to disk
    dump(result, results_path + '/hyperspace' + str(name))
