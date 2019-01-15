import os
import pickle

from hyperspace.space.robo import create_robospace
from hyperspace.space.robo import convert_robospace
from robo.fmin import bayesian_optimization

from mpi4py import MPI


def robodrive(objective, hyperparameters, results_path, n_iterations=50):
    """
    Distributed Bayesian optimization with Robo.

    Parameters:
    ----------
    * `objective` [function]:
        User defined function which calls a learner
        and returns a metric of interest.

    * `hyperparameters` [list, shape=(n_hyperparameters,)]:

   * `results_path` [string]
        Path to save optimization results

    * `n_iterations` [int, default=50]
        Number of optimization iterations
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

    robospace = create_robospace(hyperparameters)
    hyperspace = convert_robospace(robospace)
    space = hyperspace[rank]

    lower = space[0]
    upper = space[1]

    results = bayesian_optimization(objective, lower, upper, num_iterations=n_iterations)

    with open(savefile, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
