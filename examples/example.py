"""
Gradient Boosting Regressor
A hyperspace distributed version of Scikit-Optimize's hyperparameter optimization example

To Run:
mpirun -n 4 python example.py --results_dir results/skopt/space4

* Note: we use 4 processes in this example (hence -n 4 above) since we have 2**2
combinations of hyperparameter subspaces.
"""
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import argparse

from hyperspace import hyperdrive
from hyperspace.kepler import load_results


boston = load_boston()
X, y = boston.data, boston.target
n_features = X.shape[1]

reg = GradientBoostingRegressor(n_estimators=50, random_state=0)


def objective(params):
    """
    Objective function to be minimized.

    Parameters
    ----------
    * params [list, len(params)=n_hyperparameters]
        Settings of each hyperparameter for a given optimization iteration.
        - Controlled by hyperspaces's hyperdrive function.
        - Order preserved from list passed to hyperdrive's hyperparameters argument.
    """
    max_depth, learning_rate = params

    reg.set_params(max_depth=max_depth,
                   learning_rate=learning_rate)

    return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1,
                    scoring="neg_mean_absolute_error"))


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    hparams = [(2, 10),             # max_depth
               (10.0**-2, 10.0**0)] # learning_rate

    hyperdrive(objective=objective,
               hyperparameters=hparams,
               results_path=args.results_dir,
               checkpoints_path=args.results_dir,
               model="GP",
               n_iterations=10,
               verbose=True,
               random_state=0)


if __name__ == '__main__':
    main()
