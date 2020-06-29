import os
import argparse

from hyperspace import hyperdrive
from hyperspace.kepler import load_results

from skopt.benchmarks import branin


def run(results_dir, n_calls=200, n_runs=10):
    """Run benchmark for Branin function."""
    models = ['GP', 'RF', 'GBRT', 'Rand']
    bounds = [(-5.0, 10.0), (0.0, 15.0)]

    for model in models:
        model_dir = os.path.join(results_dir, model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        for random_state in range(n_runs):
            directory = os.path.join(model_dir, 'run' + str(random_state))

            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            checkpoint = load_results(directory)

            hyperdrive(
                branin,
                bounds,
                directory,
                model,
                n_iterations=n_calls,
                verbose=True,
                random_state=random_state,
                checkpoints_path=directory
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_calls', nargs="?", default=50, type=int, help="Number of function calls.")
    parser.add_argument('--n_runs', nargs="?", default=5, type=int, help="Number of runs.")
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    run(args.results_dir, args.n_calls, args.n_runs)


if __name__=='__main__':
    main()
