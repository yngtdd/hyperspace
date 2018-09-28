==================================
ML Example: Gradient Boosted Trees 
==================================

In quickstart guide we outlined how to setup hyperparameter optimization
for machine learning models. We have also seen a minimal example using 
the Styblinski-Tang function. Let's take a look at a complete machine 
learning example using Scikit-Learn.

Here we are going to optimize two hyperparameters for a gradient boosted 
regression tree model. Keeping it to two hyperparameters will allow us to
easily optimize the model on a single machine, since most processors these 
days have four cores and we can distribute the MPI ranks across those. If 
you are running a dual core machine, or would like to make the computation
a bit lighter, you can remove one of the hyperparameters, in which case you
will just have just two MPI ranks. 

We will be using the Boston Housing Dataset as it is small and readily
available through Scikit-Learn. This dataset consists of 506 samples and
13 attributes, some numeric, some categorical. The label is the median home
value for these various Boston houses. Thus, we are looking at a regression 
problem.

Gradient boosted regression trees have separate hyperparameters. For this
example we are going to optimize their `max_depth` and `learning_rate`. Our
objective function here is to minimize the negative cross validation score 
of our model over five folds of the data. The scoring metric used for the
cross-validation will be the negative mean absolute error. Alright, let's
get straight to it:

.. code-block:: python

    import argparse
    import numpy as np

    from sklearn.datasets import load_boston
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

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
        parser = argparse.ArgumentParser(description='ML Example: GBM Regression')
        parser.add_argument('--results', type=str, help='Path to results directory.')
        args = parser.parse_args()

        hparams = [(2, 10),             # max_depth
                   (10.0**-2, 10.0**0)] # learning_rate

        try:
            # Load results from previous runs
            checkpoint = load_results(args.results)
        except:
            ValueError('No prior results found. Starting from the top.')

        hyperdrive(objective=objective,
                   hyperparameters=hparams,
                   results_path=args.results,
                   model="GP",
                   n_iterations=100,
                   verbose=True,
                   random_state=0,
                   checkpoints=True,
                   restart=checkpoint)


     if __name__=='__main__':
         main()


If we save this as a module by the name gbm.py, we can run this as:

.. code-block:: console

    mpirun -n 4 python3 gbm.py --results </path/to/save/results>

You might have noticed that we added one piece to this example, the ability to load
from previous checkpoints. This is as simple as calling on the
`hyperspace.kepler.load_results` function, passing it the path where you had previously
saved your optimization results, and storing that object into a variable (here called
`checkpoint`). This will be a list of Scipy OptimizeResult objects, which contain all
information about previous runs. Then we simply pass that list to the `hyperdrive` as 
the expected value for the parameter `restart`. And voila, HyperSpace will pick up 
where it left off!

Check out the other parameters available in Scikit-learn's documentation. 
See if by including more hyperparameters you can get a better result! And if anyone is
interested, we can start up a leaderboard on our GitHub page to see who can get the best 
score. If you are interested, let me know in the GitHub issues!
