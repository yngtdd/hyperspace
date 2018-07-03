import argparse
import numpy as np

from hyperspace import robodrive
from hyperspace.benchmarks import StyblinskiTang 


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--ndims', type=int, help='Number of dimensions for Styblinski-Tang function.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    stybtang = StyblinskiTang(args.ndims)
    bounds = np.tile((-5., 5.), (args.ndims, 1))
    robodrive(stybtang, bounds, args.results_dir, n_iterations=10)


if __name__=='__main__':
    main()
