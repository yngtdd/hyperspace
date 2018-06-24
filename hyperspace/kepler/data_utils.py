import os
import re
from skopt import load


def load_results(results_path, sort=False, reverse_sort=False):
    """
    Loads results from distributed run.

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
    files = []
    for file in os.listdir(results_path):
        # Sort files by MPI rank: used for checkpointing.
        print(f'filename = {file}')
        files.append(file)
        files = sorted(files)

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

    print(f"Number of results: {len(results)}\n")
    return results, ranks
