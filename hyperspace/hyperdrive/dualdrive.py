from hyperspace.space import create_hyperspace

from skopt import gp_minimize
from skopt import gbrt_minimize
from skopt import forest_minimize
from skopt import dummy_minimize
from skopt.callbacks import DeadlineStopper
from skopt import dump

from mpi4py import MPI


def dualdrive(objective, hyperparameters, results_path, model="GP", n_iterations=50,
              verbose=False, deadline=None, random_state=0):
    """
    Distributed optimization - two optimizations per node.

    Parameters
    ----------
    * `objective` [function]:
        User defined function which calls a learner
        and returns a metric of interest.

    * `hyperparameters` [list, shape=(n_hyperparameters,)]:

    * `results_path` [string]
        Path to save optimization results

    * `model` [string, default="GP"]
        Probilistic learner used to model our objective function.
        Options:
        - "GP": Gaussian process
        - "RF": Random forest
        - "GBRT": Gradient boosted regression trees
        - "RAND": Random search

    * `n_iterations` [int, default=50]
        Number of optimization iterations

    * `verbose` [bool, default=False]
        Verbosity of optimization.

    * `deadline` [int, optional]
        Deadline (seconds) for the optimization to finish within.

    * `random_state` [int, default=0]
        Random state for reproducibility.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    if rank == 0:
        hyperspace = create_hyperspace(hyperparameters)
        hyperspace = [hyperspace[i:i+2] for i in range(0, len(hyperspace), 2)]
    else:
        hyperspace = None

    spaces = comm.scatter(hyperspace, root=0)
    space0 = spaces[0]
    space1 = spaces[1]

    if deadline:
        deadline = DeadlineStopper(deadline)

    # Thanks Guido for refusing to believe in switch statements.
    # Case 0
    if model == "GP":
        # Verbose mode should only run on node 0.
        if verbose and rank == 0:
            result0 = gp_minimize(objective, space0, n_calls=n_iterations, verbose=verbose,
                                  callback=deadline, random_state=random_state)
        # Also run on the second space 
        result1 = gp_minimize(objective, space1, n_calls=n_iterations,
                             callback=deadline, random_state=random_state)
    # Case 1
    elif model == "RF":
        if verbose and rank == 0:
            result0 = forest_minimize(objective, space0, n_calls=n_iterations, verbose=verbose,
                                      callback=deadline, random_state=random_state)
            
        result1 = forest_minimize(objective, space1, n_calls=n_iterations,
                                  callback=deadline, random_state=random_state)
    # Case 2
    elif model == "GRBRT":
        if verbose and rank == 0:
            result0 = gbrt_minimize(objective, space0, n_calls=n_iterations, verbose=verbose,
                                    callback=deadline, random_state=random_state)

        result1 = gbrt_minimize(objective, space1, n_calls=n_iterations,
                                callback=deadline, random_state=random_state)
    # Case 3
    elif model == "RAND":
        if verbose and rank == 0:
            result0 = dummy_minimize(objective, space, n_calls=n_iterations, verbose=verbose,
                                     callback=deadline, random_state=random_state)
            
        result1 = dummy_minimize(objective, space, n_calls=n_iterations,
                                 callback=deadline, random_state=random_state)
    else:
        raise ValueError("Invalid model {}. Read the documentation for "
                         "supported models.".format(model))

    # Each worker will independently write their results to disk
    dump(result0, results_path + '/hyperspace0' + str(rank))
    dump(result1, results_path + '/hyperspace1' + str(rank))


