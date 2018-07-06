import os
import re

import pickle
from skopt import load

import numpy as np
from scipy.optimize import OptimizeResult


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
