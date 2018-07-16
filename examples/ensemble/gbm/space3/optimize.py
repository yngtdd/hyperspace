import os
import argparse

import random
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import GradientBoostingClassifier

from mpi4py import MPI
from hyperspace import hyperdrive


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def load_data(val_size, test_size, seed=1):
    """
    Split data into train, val, and test.
    """
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=val_size, random_state=seed)

    return X_train, X_val, X_test, y_train, y_val, y_test


class Log:
    
    def __init__(self, colnames, savepath, rank):
        self.colnames = colnames
        self.log = pd.DataFrame(columns=self.colnames)
        self.savepath = savepath
        self.rank = rank
        self.iter = 0

    def update(self, name, acc_train, acc_val, ll):
        name = name + str(self.iter)
        entry = pd.DataFrame([[name, acc_train*100, acc_val*100, ll]], columns=self.colnames)
        self.log = self.log.append(entry)
        self.iter += 1

    def save(self):
        filename = 'log' + str(self.rank)
        logfile = os.path.join(self.savepath, filename)
        self.log.to_csv(logfile)


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
    max_depth, lr, min_impurity_decrease = params

    clf = GradientBoostingClassifier(
      max_depth=max_depth, 
      learning_rate=lr,
      min_impurity_decrease=min_impurity_decrease
    )
    
    clf.fit(X_train, y_train)
    # Training accuracy
    train_preds = clf.predict(X_train)
    acc_train = accuracy_score(y_train, train_preds)
    # Validatin accuracy
    val_preds = clf.predict(X_val)
    acc_val = accuracy_score(y_val, val_preds)
    # Validation log loss
    val_proba = clf.predict_proba(X_val)
    ll = log_loss(y_val, val_proba)
    
    name = clf.__class__.__name__
    logger.update(name, acc_train, acc_val, ll)

    return ll


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Path to save logs')
    args = parser.parse_args()

    # Logging for Visual Comparison
    global logger
    log_cols=["Classifier", "Train Accuracy", "Val Accuracy", "Log Loss"]
    logger = Log(colnames=log_cols, savepath=args.log_dir, rank=rank)

    global X_train; global X_val; global X_test; global y_train; global y_val; global y_test
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(0.25, 0.25)

    hparams = [(2, 10),             # max_depth
               (10.0**-2, 10.0**0), # learning_rate
               (0.0, 0.75)]         # min_impurity_decrease

    hyperdrive(objective=objective,
               hyperparameters=hparams,
               results_path=args.results_dir,
               model="GP",
               n_iterations=50,
               verbose=True,
               random_state=0,
               checkpoints=True)

    # Save the log data frame
    logger.save()


if __name__ == '__main__':
    main()
