import os
import re
import itertools

import pickle
from skopt import load

import numpy as np
from scipy.optimize import OptimizeResult


def _load_checkpoint(results_path, rank):
    """
    Loads checkpoint to resume optimization.

    * `results_path` [str]
        Path to the previously saved results.

    * `rank` [int]
        Rank to which the saved results belong.
    """
    files = _listfiles(results_path)
    for file in files:
        saved_rank = re.findall(r'\d+', file)
        if rank == int(saved_rank[0]):
            filepath = os.path.join(results_path, file)
            checkpoint = load(str(filepath))
            print(f'loading checkpoint for rank {int(saved_rank[0])}')
            return checkpoint


def load_results(results_path, sort=False, reverse_sort=False):
    """
    Loads results from distributed run with Scikit-Optimize.

    Parameters
    ----------
    * `results_path` [string]
        Path where results from the distributed run is stored.

    * `sort` [Bool, default=False]
        Sorts results by objective function minimum (lowest first).

    * `reverse_sort` [Bool, defaul=False]
        Sort results by objective function minimum (highest first.)
        - `sort` must be set to True.

    Returns
    -------
    * results [list]
    """
    files = _listfiles(results_path)

    ranks = []
    results = []
    for file in files:
        rank = re.findall(r'\d+', file)
        rank = int(rank[0])
        print(f'Rank: {rank}')
        ranks.append(rank)
        full_path = os.path.join(results_path, file)
        results.append(load(str(full_path)))

    if reverse_sort and not sort:
        sort = True

    if sort:
        results = sorted(results, key=lambda result: result.fun)

    return results


def load_roboresults(results_path, sort=False):
    """
    Loads results from distributed run with RoBO.

    Parameters
    ----------
    * `results_path` [string]
        Path where results from the distributed run is stored.

    * `sort` [Bool, default=False]
        Sorts results by objective function minimum (lowest first).

    Returns
    -------
    * results [list]
    """
    spacenames = _listfiles(results_path)

    results = []
    for file in spacenames:
        filepath = os.path.join(results_path, file)
        with open(filepath, 'rb') as handle:
            result = pickle.load(handle)
            results.append(result)

    if sort:
        results = sorted(results, key=lambda result: result['f_opt'])

    results = convert_roboresults(results)
    return results


def _listfiles(results_path):
    """
    Creates a list of result files names.

    Parameters:
    ----------
    * `results_path`: [str]
        Path to the saved results.
    """
    files = []
    for file in os.listdir(results_path):
        # Sort files by MPI rank: used for checkpointing.
        files.append(file)

    files = sorted(files)
    return files


def _convert_robo(result):
    """
    Convert results from RoBO to scipy.OptimizeResult.

    Parameters:
    ----------
    * `result`: [dict]
      Result from optimization when using RoBO.

    Returns:
    -------
    * `optresult`: [scipy.optimize.OptimizeResult]
      Result formatted to match Scikit-Optimize.
    """
    # Attributes that match Scikit-optimize
    x = result['x_opt']
    fun = result['f_opt']
    func_vals = np.array(result['y'])
    x_iters = result['X']

    # information particular to RoBO
    specs = {
      'incumbents': result['incumbents'],
      'incumbent_values': result['incumbent_values'],
      'runtime': result['runtime'],
      'overhead': result['overhead']
    }

    optresult = OptimizeResult(
      x=x,
      fun=fun,
      func_vals=func_vals,
      x_iters=x_iters,
      specs=specs
    )

    return optresult


def convert_roboresults(results):
    """
    Convert results from RoBO to scipy.OptimizeResults.

    Parameters:
    ----------
    * `results` [list]
      Collection of results from distributed RoBO optimization.

    Results:
    -------
    list of scipy.optimize.OptimizeResults.
    """
    return [_convert_robo(x) for x in results]


def create_result(Xi, yi, n_evaluations=None, space=None, rng=None, specs=None, models=None):
    """
    Initialize an `OptimizeResult` object.

    Parameters
    ----------
    * `Xi` [list of lists, shape=(n_iters, n_features)]:
        Location of the minimum at every iteration.

    * `yi` [array-like, shape=(n_iters,)]:
        Minimum value obtained at every iteration.

    * `space` [Space instance, optional]:
        Search space.

    * `rng` [RandomState instance, optional]:
        State of the random state.

    * `specs` [dict, optional]:
        Call specifications.

    * `models` [list, optional]:
        List of fit surrogate models.

    Returns
    -------
    * `res` [`OptimizeResult`, scipy object]:
        OptimizeResult instance with the required information.
    """
    res = OptimizeResult()

    try:
        # Hyperband returns evaluations as lists of lists.
        # We want to store the results as a single array.
        yi = list(itertools.chain.from_iterable(yi))
        Xi = list(itertools.chain.from_iterable(Xi))
    except TypeError:
        # All algorithms other than Hyperband already return a single list.
        pass

    yi = np.asarray(yi)
    if np.ndim(yi) == 2:
        res.log_time = np.ravel(yi[:, 1])
        yi = np.ravel(yi[:, 0])
    best = np.argmin(yi)
    res.x = Xi[best]
    res.fun = yi[best]

    if n_evaluations:
        unique, sort_indices = np.unique(yi, return_index=True)

        if len(unique) < n_evaluations:
            func_sort_idx = np.argsort(yi)
            func_vals = sorted(yi)
            res.func_vals = np.asarray(func_vals[:n_evaluations])

            x_iter_sort = []
            for idx in func_sort_idx:
                x_iter_sort.append(Xi[idx])

            res.x_iters = np.asarray(x_iter_sort[:n_evaluations])
            res.all_func_vals = np.asarray(yi)
            res.all_x_iters = np.asarray(Xi)
        else:
            func_vals = sorted(unique)
            res.func_vals = np.asarray(func_vals[:n_evaluations])

            x_iter_sort = []
            for idx in sort_indices:
                x_iter_sort.append(Xi[idx])

            res.x_iters = np.asarray(x_iter_sort[:n_evaluations])
            res.all_func_vals = np.asarray(yi)
            res.all_x_iters = np.asarray(Xi)
    else:
        res.func_vals = np.asarray(yi)
        res.x_iters = np.asarray(Xi)

    res.models = models
    res.space = space
    res.random_state = rng
    res.specs = specs
    return res
