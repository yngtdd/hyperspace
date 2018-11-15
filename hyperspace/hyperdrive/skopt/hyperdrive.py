from hyperspace.space import create_hyperspace
from hyperspace.space import create_hyperbounds
from hyperspace.kepler import _load_checkpoint
from hyperspace.rover.checkpoints import CheckpointSaver
from hyperspace.rover.latin_hypercube_sampler import lhs_start

from skopt import gp_minimize
from skopt import gbrt_minimize
from skopt import forest_minimize
from skopt import dummy_minimize
from skopt.callbacks import DeadlineStopper
from skopt import dump

import os
from mpi4py import MPI


def hyperdrive(objective, hyperparameters, results_path, model="GP", n_iterations=50, verbose=False,
               checkpoints_path=None, deadline=None, sampler=None, n_samples=None, random_state=0):
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

    * `checkpoint_path` [string]
        Path to previously saved results. Used to resume optimization.

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

    * `checkpoints` [bool, default=False]
        Whether to checkpoint at each step of the optimization.

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

    if checkpoints_path and sampler:
        raise ValueError('Cannot use both a restart from a previous run and ' \
                         'use latin hypercube sampling for initial search points!')

    # Setup savefile
    if rank < 10:
        # Ensure results are sorted by rank
        filename = 'hyperspace' + str(0) + str(rank)
    else:
        filename = 'hyperspace' + str(rank)

    savefile = os.path.join(results_path, filename)

    # Create hyperspaces, and either sampling bounds or checkpoints
    hyperspace = create_hyperspace(hyperparameters)
    space = hyperspace[rank]

    # Latin hypercube sampling
    if sampler and not n_samples:
        raise ValueError(f'Sampler requires n_samples > 0. Got {n_samples}')
    elif sampler and n_samples:
        hyperbounds = create_hyperbounds(hyperparameters)
        bounds = hyperbounds[rank]
        # Get initial points in domain via latin hypercube sampling
        init_points = lhs_start(bounds, n_samples)
        init_response = None
        n_rand = 10 - len(init_points)
    else:
        init_points = None
        init_response = None
        n_rand = 10

    # Resuming from checkpoint
    if checkpoints_path:
        checkpoint = _load_checkpoint(checkpoints_path, rank)
        try:
            init_points = checkpoint.x_iters
            init_response = checkpoint.func_vals
            n_rand = 10 - len(init_points)
        except AttributeError:
            # Missing saves won't have initial values.
            init_points = None
            init_response = None
            n_rand = 10

    callbacks = []
    if deadline:
        deadline = DeadlineStopper(deadline)
        callbacks.append(deadline)

    if checkpoints_path:
        checkpoint_callback = CheckpointSaver(checkpoints_path, filename)
        callbacks.append(checkpoint_callback)

    # Thanks Guido for refusing to believe in switch statements.
    # Case 0
    if model == "GP":
        # Verbose mode should only run on node 0.
        if verbose and rank == 0:
            result = gp_minimize(objective, space, n_calls=n_iterations, verbose=verbose,
                                 callback=callbacks, x0=init_points, y0=init_response,
                                 n_random_starts=n_rand, random_state=random_state)
        else:
            result = gp_minimize(objective, space, n_calls=n_iterations,
                                 callback=callbacks, x0=init_points, y0=init_response,
                                 n_random_starts=n_rand, random_state=random_state)

    # Case 1
    elif model == "RF":
        if verbose and rank == 0:
            result = forest_minimize(objective, space, n_calls=n_iterations, verbose=verbose,
                                     callback=callbacks, x0=init_points, y0=init_response,
                                     n_random_starts=n_rand, random_state=random_state)
        else:
            result = forest_minimize(objective, space, n_calls=n_iterations,
                                     callback=callbacks, x0=init_points, y0=init_response,
                                     n_random_starts=n_rand, random_state=random_state)
    # Case 2
    elif model == "GBRT":
        if verbose and rank == 0:
            result = gbrt_minimize(objective, space, n_calls=n_iterations, verbose=verbose,
                                   callback=callbacks, x0=init_points, y0=init_response,
                                   n_random_starts=n_rand, random_state=random_state)
        else:
            result = gbrt_minimize(objective, space, n_calls=n_iterations,
                                   callback=callbacks, x0=init_points, y0=init_response,
                                   n_random_starts=n_rand, random_state=random_state)
    # Case 3
    elif model == "RAND":
        if verbose and rank == 0:
            result = dummy_minimize(objective, space, n_calls=n_iterations, verbose=verbose,
                                    callback=callbacks, x0=init_points, y0=init_response,
                                    random_state=random_state)
        else:
            result = dummy_minimize(objective, space, n_calls=n_iterations,
                                    callback=callbacks, x0=init_points, y0=init_response,
                                    random_state=random_state)
    else:
        raise ValueError("Invalid model {}. Read the documentation for "
                         "supported models.".format(model))

    # Each worker will independently write their results to disk
    dump(result, savefile)
