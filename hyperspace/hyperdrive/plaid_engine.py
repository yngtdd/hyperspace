import sys
from mpi4py import MPI
from collections import deque

from skopt import dump
from engine_models import minimize

TAG_WORKER_FINISHED = 20
TAG_RAW_DATA = 10
TAG_SETUP = 0
TAG_KILL = 5


def control(comm, rank, nprocs, hyperspace, hyperbounds=None):
    """
    Orchestrates all distributed runs.
    """
    num_workers = nprocs - 1
    worker_queue = deque()
    space_queue = deque()
    space_queue.extend(hyperspace)

    if hyperbounds:
        bounds_queue = deque()
        bounds_queue.extend(hyperbounds)

    for i in range(1, nprocs):
        comm.send("START", dest=i, tag=TAG_SETUP)
        print("master sending START to ", str(i))

    while len(worker_queue) < num_workers:
        worker_rank = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        print("control received OK from ", worker_rank)
        worker_queue.append(worker_rank)
        sys.stdout.flush()

    #print('worker_queue now has {} workers'.format(len(worker_queue)))

    stillWorking = True
    currentJobs = {}

    workers_finished = 0
    while stillWorking: # or (len(space) > 0):
        msg = comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        if msg:
            status = MPI.Status()
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if status.tag == TAG_WORKER_FINISHED:
                # a worker has finished, do something
                print('Satelite {} has finished!'.format(status.source))
                workers_finished += 1

                worker_queue.append(status.source)
                #print('Appending rank {} to worker_queue'.format(status.source))
                #print('worker_queue now has {} workers'.format(len(worker_queue)))
                sys.stdout.flush()

        if len(worker_queue) > 0:
            # Get the next worker ID
            worker_rank = worker_queue.pop()
            try:
                # Get the next hyperspace from queue
                space = space_queue.pop()
                comm.send(space, dest=worker_rank, tag=TAG_RAW_DATA)
                print("Sent space {}".format(space))

                if hyperbounds:
                    bounds = bounds_queue.pop()
                    comm.send(bounds, dest=worker_rank, tag=TAG_RAW_DATA)
                    print("Sent bounds {}".format(bounds))
            except IndexError:
                # There is no longer any hyperspace to be searched over.
                stillWorking = False

    # KILL ALL PROCESSES WHEN DONE
    for i in range(1, nprocs):
        comm.send("KILL", dest=i, tag=TAG_KILL)
        print("master sending KILL to {}".format(i))
    print('Number of workers finished: {}'.format(workers_finished))


def satelites(comm, rank, objective, model, n_iterations,
              results_path, verbose, deadline, random_state):
    """
    Distributed worker at each MPI rank.

    Parameters
    ----------
    * `comm` [mpi4py communicator]:
        Handles communication between ranks.

    * `rank` [int]:
        MPI rank.

    * `objective` [function]:
        User defined function which calls a learner
        and returns a metric of interest.

    * `results_path` [string]
        Path to save optimization results

    * `model` [string, default="GP"]
        Probilistic learner used to model our objective function.
        Options:
        - "GP": Gaussian process
        - "RF": Random forest
        - "GBRT": Gradient boosted regression trees
        - "RAND": Random search

    * `n_iterations` [int, default=50]
        Number of optimization iterations

    * `verbose` [bool, default=False]
        Verbosity of optimization.

    * `deadline` [int, optional]
        Deadline (seconds) for the optimization to finish within.

    * `random_state` [int, default=0]
        Random state for reproducibility.
    """
    data = comm.recv(source=0, tag=TAG_SETUP)
    print('hello from rank {}! I have data {}'.format(rank, data))
    # send startup messages
    comm.send(rank, dest=0)

    stillWorking = True
    while stillWorking:
        msg = comm.Iprobe(source=0, tag=MPI.ANY_TAG)
        if msg:
            status = MPI.Status()
            space = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

            result = minimize(objective=objective, space=space, rank=rank,
                              results_path=results_path, model=model,
                              n_iterations=n_iterations, verbose=verbose,
                              deadline=deadline, random_state=random_state):

            comm.send(result, dest=0, tag=TAG_WORKER_FINISHED)
            if status.tag == TAG_KILL:
                stillWorking = False
