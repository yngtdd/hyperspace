"""Latin Hypercube Sampling"""
import numbers
import random
import numpy as np


def sample_latin_hypercube(low, high, n_samples, n_dims=None, rng=None):
    """
    Creates initial design of n_samples drawn from a latin hypercube.

    Parameters:
    ----------
    * `low`: [np.array, shape=(n_dims,)]
        lower bound for each dimension to be sampled.

    * `high`: [np.array, shape=(n_dims,)]
        upper bound for each dimension to be sampled.

    * `n_samples`: [int]
        number of samples to be drawn.
        - Must be < number of unique points within each dimension's bounds.

    Returns:
    -------
    * `samples.T`: [np.array, shape=(n_samples, n_dims)]
    """
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))
    if n_dims is None:
        n_dims = low.shape[0]

    samples = []
    for i in range(n_dims):
        if isinstance(low[i], numbers.Integral):
            sample = random.sample(range(low[i], high[i]), n_samples)
        elif isinstance(low[i], numbers.Real):
            lower_bound = low[i]
            upper_bound = high[i]
            sample = lower_bound + rng.uniform(0, 1, n_samples) * (upper_bound - lower_bound)
        else:
            raise ValueError('Latin hypercube sampling can only draw from types int and real,'
                             ' got {}!'.format(type(low[i])))

        samples.append(sample)

    samples = np.array(samples, dtype=object)

    for i in range(n_dims):
        rng.shuffle(samples[i, :])

    return samples.T


def lhs_start(hyperbounds, n_samples, rng=None):
    """
    Creates the initial search space using latin hypercube sampling.
    """
    low_bounds = []
    high_bounds = []
    for bound in hyperbounds:
        low_bounds.append(bound[0])
        high_bounds.append(bound[1])

    n_dims = len(hyperbounds)
    samples = sample_latin_hypercube(low_bounds, high_bounds, n_samples, n_dims=n_dims, rng=rng)
    samples = samples.tolist()
    return samples
