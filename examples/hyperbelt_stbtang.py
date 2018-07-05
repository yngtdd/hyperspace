"""
Distributed Hyperband with SMBO. 

We take the Hyperband algorithm, replace the random sampling with Bayesian
optimization using a Gaussian process, and run in parallel according to the
HyperSpace algorithm.

Usage:
mpirun -n 32 python hyperbelt.py --results_dir ./results/hyperbelt
"""
import argparse
import numpy as np

from hyperspace import hyperbelt 
from hyperspace.benchmarks import StyblinskiTang


stybtang = StyblinskiTang(5)


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    bounds = np.tile((-5., 5.), (5, 1))

    hyperbelt(objective=stybtang,
              hyperparameters=bounds,
              results_path=args.results_dir,
              model="RAND",
              n_iterations=50,
#              sampler='lhs',
#              n_samples=2,
              model_verbose=False,
              hyperband_verbose=True,
              random_state=0)


if __name__ == '__main__':
    main()

