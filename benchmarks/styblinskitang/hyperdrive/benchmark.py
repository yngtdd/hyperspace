import argparse
import numpy as np

from hyperspace import hyperdrive
from hyperspace.benchmarks import StyblinskiTang


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--ndims', type=int, help='Number of dimensions for Styblinski-Tang function')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    stybtang = StyblinskiTang(args.ndims)
    bounds = np.tile((-5., 5.), (args.ndims, 1))

    hyperdrive(objective=stybtang,
              hyperparameters=bounds,
              results_path=args.results_dir,
              checkpoints_path=args.results_dir,
              model="GP",
              n_iterations=20,
              verbose=True,
              random_state=0)


if __name__ == '__main__':
    main()
