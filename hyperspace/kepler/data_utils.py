import os
import re

import pickle
from skopt import load


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
        print(f'full path is {full_path} of type {type(full_path)}')
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
