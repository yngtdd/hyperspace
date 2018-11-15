"""
Distributed Hyperband with SMBO.

We take the Hyperband algorithm, replace the random sampling with Bayesian
optimization using a Gaussian process, and run in parallel according to the
HyperSpace algorithm.

Usage:
mpirun -n 8 python hyperbelt.py --results_dir ./results/hyperbelt
"""
import os
import argparse
import itertools
import numpy as np

import skopt
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from hyperspace import hyperband


boston = load_boston()
X, y = boston.data, boston.target
n_features = X.shape[1]

reg = GradientBoostingRegressor(n_estimators=50, random_state=0)


def objective(params, iterations):
    """
    Objective function to be minimized.

    Parameters
    ----------
    * params [list, len(params)=n_hyperparameters]
        Settings of each hyperparameter for a given optimization iteration.
        - Controlled by hyperspaces's hyperdrive function.
        - Order preserved from list passed to hyperdrive's hyperparameters argument.
    """
    max_depth, learning_rate, max_features = params

    reg.set_params(max_depth=max_depth,
                   learning_rate=learning_rate,
                   max_features=max_features)

    return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1,
                                    scoring="neg_mean_absolute_error"))


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    hparams = [(2, 10),             # max_depth
               (10.0**-2, 10.0**0), # learning_rate
               (1, 10)]             # max_features

    res = hyperband(objective, hparams, max_iter=100, eta=3, verbose=True, random_state=0)
    results_path = os.path.join(args.results_dir, 'hyperband_gbm.pkl')
    skopt.dump(res, results_path)

if __name__ == '__main__':
    main()

