import argparse
import numpy as np

from optimize import load_data, objective
from hyperspace.kepler import load_results

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def ensemble(clfs, X_train, y_train, X_test, y_test, nfolds=5):
    """
    Train an ensemble of HyperSpace models

    Parameters:
    ----------
    * `clfs` [list]
      Classifiers to be ensembled

    * `X_train` [ndarray]
      Training set features

    * `y_train` [ndarray]
      Training set labels

    * `X_test` [ndarray]
      Test set features

    * `y_test` [ndarray]
      Test set labels
      - Used to get initial estimate of each model.

    * `nfolds` [int]
      Number of cross-validation folds.
      More folds means more features for ensemble blend.

    Returns:
    -------
    * `y_preds` [ndarray]
      Class predictions for the test set.

    * `y_proba` [ndarray]
      Predicted probabilities for the test set.
    """
    # Use stratified k-fold validation to get multiple multiple estimates from each clf.
    skf = StratifiedKFold(n_splits=nfolds)
    folds = list(skf.split(X_train, y_train))

    dataset_blend_train = np.zeros((X_train.shape[0], nfolds))
    dataset_blend_test = np.zeros((X_test.shape[0], nfolds))
   
    for iteration, clf in enumerate(clfs):
        print('\n' + '=' * 15)
        print(f'Iteration {iteration}, Random Forest Classifier') 
        print(f'Hyperparameters: n_trees: {clf.n_estimators}, max_depth: {clf.max_depth}')

        # See how individual models perfom on test set.
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        model_acc_test = accuracy_score(y_test, preds)
        print(f'Test Set accuracy: {model_acc_test:.4f}')

        print(f'Beginning cross-valiation for model ensemble.')
        dataset_blend_test_iter = np.zeros((X_test.shape[0], nfolds))
        for fold, (train, val) in enumerate(folds):
            print(f'==> Fitting on fold: {fold}') 
            X_train_fold = X_train[train]
            y_train_fold = y_train[train]
            X_val = X_train[val]
            y_val = y_train[val]
            clf.fit(X_train_fold, y_train_fold)
            y_submission = clf.predict_proba(X_val)[:, 1]
            dataset_blend_train[val, iteration] = y_submission
            dataset_blend_test_iter[:, fold] = clf.predict_proba(X_test)[:, 1]
        dataset_blend_test[:, iteration] = dataset_blend_test_iter.mean(1)
        print(f'=' * 15)

    print('\nBlending predictions with logistic regression')
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y_train)
    y_preds = clf.predict(dataset_blend_test)
    y_proba = clf.predict_proba(dataset_blend_test)[:, 1]

    # Normalizing our predicted probabilities.
    y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

    print('Saving Ensemble Predicted Probabilities.')
    tmp = np.vstack([range(1, len(y_proba)+1), y_proba]).T
    np.savetxt(
      fname='baseline_submission.csv', X=tmp, fmt='%d,%0.9f',
      header='TestID, PredictedProbability', comments=''
    )

    return y_preds, y_proba


def main():
    parser = argparse.ArgumentParser(description='Ensembling HyperSpace models')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    results = load_results(args.results_dir)

    clfs = []
    for _ in results:
        # Use default random forest for each of the ensemble models
        model = RandomForestClassifier()
        clfs.append(model)

    # Using the training and test sets from `optimize.py`
    X_train, _, X_test, y_train, _, y_test = load_data(0.25, 0.25)
    y_preds, y_proba = ensemble(clfs, X_train, y_train, X_test, y_test)

    acc_test = accuracy_score(y_test, y_preds)
    ll_test = log_loss(y_test, y_proba)

    print(f'\nEnsemble Test Accuracy: {acc_test:.4f}, Test Log Loss: {ll_test:.4f}')


if __name__=='__main__':
    main()
