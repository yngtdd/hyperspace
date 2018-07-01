import numpy as np

from hyperspace.benchmarks import StyblinksiTang 
from hyperspace.hyperdrive.robo.hyperbayes import robodrive


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    stybtang = StyblinksiTang(2)
    bounds = np.tile((-5., 5.), (2, 1))
    robodrive(stybtang, bounds, args.results_path, n_iterations=10)


if __name__=='__main__':
    main()
