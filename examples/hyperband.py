"""
Original Hyperband algorithm.
"""
import os
import argparse
import numpy as np
from skopt import dump

from hyperspace import hyperband
from hyperspace.benchmarks import StyblinskiTang


stybtang = StyblinskiTang(5)


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    bounds = np.tile((-5., 5.), (5, 1))

    results = hyperband(objective=stybtang,
                        space=bounds,
                        model="RAND",
                        model_verbose=True,
                        hyperband_verbose=True,
                        random_state=0)

    results_path = os.path.join(args.results_dir, 'hyperband_stybtang.pkl')
    dump(results, results_path)


if __name__ == '__main__':
    main()

