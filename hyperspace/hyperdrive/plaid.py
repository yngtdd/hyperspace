from mpi4py import MPI

from plaid_engine import control, satelites
from hyperspace.space import create_hyperspace
from hyperspace.space import create_hyperbounds


def hyperdrive(objective, hyperparameters, results_path, model="GP", n_iterations=50,
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
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        hyperspace = create_hyperspace(hyperparameters)

        if sampler and not n_samples:
            raise ValueError('Sampler requires n_samples > 0. Got {}'.format(n_samples))
        elif sampler and n_samples:
            hyperbounds = create_hyperbounds(hyperparameters)
        else:
            hyperbounds = None

        control(comm=comm, rank=0, nprocs=nprocs, hyperspace=hyperspace, hyperbounds=hyperbounds)
        print("Total Time: ",  time.time()-start_time)

    else:
        satelites(comm=comm, rank=rank, objective=objective, model=model,
                  n_iterations=n_iterations, results_path=results_path,
                  verbose=verbose, deadline=deadline, random_state=random_state)
