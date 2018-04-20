from hyperspace.hyperdrive import hyperband
from hyperspace.space import create_hyperspace
from hyperspace.space import create_hyperbounds
from hyperspace.rover.latin_hypercube_sampler import lhs_start

from skopt.callbacks import DeadlineStopper
from skopt import dump

from mpi4py import MPI


def hyperdrive(objective, hyperparameters, results_path, model="RAND", n_iterations=50,
               verbose=False, deadline=None, sampler=None, n_samples=None, random_state=0):
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

    * `deadline` [int, optional]
        Deadline (seconds) for the optimization to finish within.

    * `sampler` [str, default=None]
        Random sampling scheme for optimizer's initial runs.
        Options:
        - "lhs": latin hypercube sampling

    * `n_samples` [int, default=None]
        Number of random samples to be drawn from the `sampler`.
        - Required if you would like to use `sampler`.
        - Must be <= the number of elements in the smallest hyperparameter bound's set.

    * `random_state` [int, default=0]
        Random state for reproducibility.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    if rank == 0:
        hyperspace = create_hyperspace(hyperparameters)

        if sampler and not n_samples:
            raise ValueError('Sampler requires n_samples > 0. Got {}'.format(n_samples))
        elif sampler and n_samples:
            hyperbounds = create_hyperbounds(hyperparameters)
    else:
        hyperspace = None
        if sampler is not None:
            hyperbounds = None

    space = comm.scatter(hyperspace, root=0)

    if sampler:
        bounds = comm.scatter(hyperbounds, root=0)
        # Get initial points in the obj. function domain via latin hypercube sampling
        init_points = lhs_start(bounds, n_samples)
        n_rand = 10 - len(init_points)
    else:
        init_points = None
        n_rand = 10

    if deadline:
        deadline = DeadlineStopper(deadline)

    result = hyperband(objective, space, model='RAND', x_init=init_points, n_random_starts=n_rand,
    # Each worker will independently write their results to disk
    dump(result, results_path + '/hyperspace' + str(rank))
