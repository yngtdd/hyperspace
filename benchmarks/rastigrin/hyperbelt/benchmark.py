import argparse
import numpy as np

from hyperspace import hyperbelt 
from hyperspace.benchmarks import Rastigrin 


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--ndims', type=int, help='Number of dimensions for Rastigrin function')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    rastigrin = Rastigrin(args.ndims)
    bounds = np.tile((-5.12, 5.12), (args.ndims, 1))

    hyperbelt(objective=rastigrin,
              hyperparameters=bounds,
              results_path=args.results_dir,
              model="RAND",
              n_iterations=50,
              sampler='lhs',
              n_samples=2,
              model_verbose=False,
              hyperband_verbose=True,
              random_state=0)


if __name__ == '__main__':
    main()

