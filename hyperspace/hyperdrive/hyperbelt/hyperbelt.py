from hyperspace.space import create_hyperspace
from hyperspace.space import create_hyperbounds
from hyperspace.rover.latin_hypercube_sampler import lhs_start
from hyperspace.hyperdrive.hyperbelt.hyperband import hyperband

from skopt.callbacks import DeadlineStopper
from skopt import dump

import os
from mpi4py import MPI


def hyperbelt(objective, hyperparameters, results_path, max_iter=100, eta=3,
              verbose=True, n_evaluations=None, random_state=0):
    """
    Distributed HyperBand with SMBO - one hyperspace per node.

    Parameters
    ----------
    * `objective` [function]:
        User defined function which calls a learner
        and returns a metric of interest.

    * `hyperparameters` [list, shape=(n_hyperparameters,)]:

    * `results_path` [string]
        Path to save optimization results

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

    # Setup savefile
    if rank < 10:
        # Ensure results are sorted by rank
        filename = 'hyperspace' + str(0) + str(rank)
    else:
        filename = 'hyperspace' + str(rank)

    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    savefile = os.path.join(results_path, filename)

    hyperspace = create_hyperspace(hyperparameters)
    space = hyperspace[rank]

    result = hyperband(objective, space, max_iter, eta,
                       random_state, verbose, n_evaluations, rank)

    # Each worker will independently write their results to disk
    dump(result, savefile)
