from gbm import objective
from hyperspace.kepler import load_results

from mpi4py import MPI


def return_to_hyperspace(results_dir):
    """
    Return to Hyperspace to profile best iteration.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        results = load_results(results_dir)
        rank_domains = [result.x for result in results]
    else:
        rank_domains = None

    domain = comm.scatter(rank_domains, root=0)
    cv_score = objective(domain)
    print('Cross validation score at rank {} is {}'.format(rank, cv_score))


def main():
    results_dir = '/Users/youngtodd/hyperspace/examples/gbm_results'
    # profile start
    return_to_hyperspace(results_dir)
    # profile stop


if __name__ =='__main__':
    main()
