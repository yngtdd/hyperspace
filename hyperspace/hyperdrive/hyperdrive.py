from ..space import create_hyperspace

from skopt import gp_minimize
from skopt import gbrt_minimize
from skopt import forest_minimize
from skopt import dummy_minimize
from skopt import dump

from mpi4py import MPI


def hyperdrive(objective, hyperparameters, results_path, model="GP", n_iterations=50,
               verbose=False, random_state=0):
    """
    Distributed optimization - one optimization per node.

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

    * `random_state` [int, default=0]
        Random state for reproducibility.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    if rank == 0:
        hyperspace = create_hyperspace(hyperparameters)
    else:
        hyperspace = None

    space = comm.scatter(hyperspace, root=0)

    # Thanks Guido for refusing to believe in switch statements.
    # Case 0
    if model == "GP":
        # Verbose mode should only run on node 0.
        if verbose and rank == 0:
            result = gp_minimize(objective, space, n_calls=n_iterations, verbose=verbose,
                                 random_state=random_state)
        result = gp_minimize(objective, space, n_calls=n_iterations, random_state=random_state)
    # Case 1
    elif model == "RF":
        if verbose and rank == 0:
            result = forest_minimize(objective, space, n_calls=n_iterations, verbose=verbose,
                                     random_state=random_state)
        result = forest_minimize(objective, space, n_calls=n_iterations, random_state=random_state)
    # Case 2
    elif model == "GRBRT":
        if verbose and rank == 0:
            result = gbrt_minimize(objective, space, n_calls=n_iterations, verbose=verbose,
                                   random_state=random_state)
        result = gbrt_minimize(objective, space, n_calls=n_iterations, random_state=random_state)
    # Case 3
    elif model == "RAND":
        if verbose and rank == 0:
            result = dummy_minimize(objective, space, n_calls=n_iterations, verbose=verbose,
                                    random_state=random_state)
        result = dummy_minimize(objective, space, n_calls=n_iterations, random_state=random_state)
    else:
        raise ValueError("Invalid model {}. Read the documentation for "
                         "supported models.".format(model))

    # Each worker will independently write their results to disk
    dump(result, results_path + 'hyperspace' + str(rank))
