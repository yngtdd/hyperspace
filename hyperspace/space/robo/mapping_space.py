import numbers
import numpy as np

from .space import RoboInteger
from .space import RoboReal


def check_robo_dimension(dimension):
    """
    Turn a provided dimension description into a dimension object.
    Checks that the provided dimension falls into one of the
    supported types. For a list of supported types, look at
    the documentation of ``dimension`` below.
    If ``dimension`` is already a ``Dimension`` instance, return it.

   Parameters
   ----------
    * `dimension`:
        Search space Dimension.
        Each search dimension can be defined either as
        - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

    Returns
    -------
    * `dimension`:
        Dimension instance.
    """
    if not isinstance(dimension, (list, tuple, np.ndarray)):
        raise ValueError("Dimension has to be a list or tuple.")

    if len(dimension) == 2:
        if all([isinstance(dim, numbers.Integral) for dim in dimension]):
            roboint = RoboInteger(*dimension)
            return roboint.get_hyperspace()
        elif any([isinstance(dim, numbers.Real) for dim in dimension]):
            roboreal = RoboReal(*dimension)
            return roboreal.get_hyperspace()
        else:
            raise ValueError(f"Invalid dimension {dimension}. " \
                             f"Read the documentation for supported types.")

    if len(dimension) > 2:
        raise ValueError(f"Cannot have a dimension with more than two bounds and "\
                         f"Robo does not support categorical variales!")

    raise ValueError(f"Invalid dimension {dimension}. Read the documentation for supported types.")


def fold_spaces(low_spaces, high_spaces):
    """
    Creates all possible combinations of hyperspaces.

    Parameters
    ----------
    * `low_spaces` [list, shape=(n_spaces,)]:
        lower spaces defined by hyperspace classes.

    * `high_spaces` [list, shape=(n_spaces,)]:
        lower spaces defined by hyperspace classes.

    Returns
    -------
    * `hyperspace` [`list of lists`, shape=(2**n_spaces, n_spaces)]:
        - All combinations of hyperspaces. Each list within hyperspace
          is a search space to be distributed across 2**n_spaces nodes.
    """
    if len(low_spaces)  != len(high_spaces):
        raise ValueError(("low_spaces and high_spaces must have the same length. "
                         "Got {} and {} respectively.".format(len(low_spaces), len(high_spaces))))

    indices = len(low_spaces)
    num_hyperspaces = 2**indices
    hyperspace = [[] for i in range(num_hyperspaces)]

    for space in range(num_hyperspaces):
        for index in range(indices):
            bit_tester = 1 << index
            if space & bit_tester:
                hyperspace[space].insert(index, low_spaces[index])
            else:
                hyperspace[space].insert(index, high_spaces[index])

    return hyperspace


def create_robospace(hyperparameters):
    """
    Converts hyperparameter lists to HyperSpace robo instances.

    Parameters
    ----------
    * `hyperparameters` [list, shape=(n_hyperparameters,)]

    Returns
    -------
    * `robospace` [list of lists, shape(n_spaces, n_hyperparameters)]
        - All combinations of hyperspaces. Each list within hyperspace
          is a search space to be distributed across 2**n_spaces nodes.
    """
    hparams_low = []
    hparams_high = []
    for hparam in hyperparameters:
        low, high = check_robo_dimension(hparam)
        hparams_low.append(low)
        hparams_high.append(high)

    all_spaces = fold_spaces(hparams_low, hparams_high)

    robospace = []
    for space in all_spaces:
        robospace.append(space)

    return robospace


def convert_robospace(robospace):
    """
    Convert robospace from tuple form to array form.

    Parameters:
    ----------
    * `robospace`: [list of list of tuples, shape(n_spaces, n_hyperparameters)]

    Returns:
    -------
    * `robospaces`: [list of lists of np.arrays, shape(n_spaces, n_hyperparameters)]
    """
    robospaces = []
    for space in robospace:
        lower_bounds = []
        upper_bounds = []
        for dim in space:
            # We are guaranteed to have just two dimension bounds.
            lower_bounds.append(dim[0])
            upper_bounds.append(dim[1])

        lower_bounds = np.asarray(lower_bounds)
        upper_bounds = np.asarray(upper_bounds)
        robospaces.append([lower_bounds, upper_bounds])

    return robospaces
