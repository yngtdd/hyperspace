import warnings
from abc import ABCMeta, abstractmethod

from math import floor, ceil

from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical


class _Ellipsis:
    def __repr__(self):
        return '...'


class HyperSpace(object):
    """
    Base class for all HyperSpaces.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_hyperspace(self):
        raise NotImplementedError("You should implement this!")


class HyperInteger(HyperSpace, Integer):
    """
    HyperSpace for Integers.

    Parameters
    ----------
    * `low` [int]:
        Lower bound (inclusive).

    * `high` [int]:
        Upper bound (inclusive).

    * `transform` ["identity", "normalize", optional]:
        The following transformations are supported.

        - "identity", (default) the transformed space is the same as the
           original space.
        - "normalize", the transformed space is scaled to be between
           0 and 1.

    * `name` [str or None]:
        Name associated with dimension, e.g., "number of trees".

    * `overlap` [float, default=0.25]:
        Amount of overlap between between each hyperspace.
        - Should be between 0 and 1.
        - If overlap=0, there are no shared values between the hyperspaces.
        - If overlap=1, two copies of the search space is made.
    """
    def __init__(self, low, high, transform=None, overlap=0.25, name=None):
        super().__init__(low, high, transform)
        #self.transform = transform
        self.overlap = overlap
        self.name = name
        self.space0_low = None
        self.space0_high = None
        self.space1_low = None
        self.space1_high = None
        self._divide_space()

        if high <= low:
            raise ValueError("the lower bound {} has to be less than the"
                             " upper bound {}".format(low, high))

    def __repr__(self):
        """
        Representation of the Integer HyperSpace. Useful when checking the hyperspace bounds.
        """
        return "HyperInteger(low={}, high={})\n" \
               "HyperInteger(low={}, high={})".format(self.space0_low, self.space0_high,
                                                      self.space1_low, self.space1_high)

    def _divide_space(self):
        """
        Divides the original search space into overlapping subspaces.
        """
        subinterval_length = abs(self.high - self.low)/2
        overlap_length = subinterval_length * self.overlap

        if subinterval_length < 1:
            warnings.warn("Each hyperspace contains a single value.")

        # Define the bounds of the hyperspaces.
        # Mind the floor and ceiling: spaces defined with short ranges can get interesting.
        self.space0_low = self.low
        self.space0_high = floor(self.space0_low + subinterval_length + overlap_length)
        self.space1_low = ceil(self.high - (subinterval_length + overlap_length))
        self.space1_high = self.high

    def get_hyperspace(self):
        """
        Create integer HyperSpaces.
        """
        return Integer(self.space0_low, self.space0_high), \
               Integer(self.space1_low, self.space1_high)


class HyperReal(HyperSpace, Real):
    """
    HyperSpace for Integers.

    Parameters
    ----------
    * `low` [float]:
        Lower bound (inclusive).

    * `high` [float]:
        Upper bound (inclusive).

    * `prior` ["uniform" or "log-uniform", default="uniform"]:
            Distribution to use when sampling random points for this dimension.
            - If `"uniform"`, points are sampled uniformly between the lower
              and upper bounds.
            - If `"log-uniform"`, points are sampled uniformly between
              `log10(lower)` and `log10(upper)`.`

    * `transform` ["identity", "normalize", optional]:
        The following transformations are supported.

        - "identity", (default) the transformed space is the same as the
           original space.
        - "normalize", the transformed space is scaled to be between
           0 and 1.

    * `name` [str or None]:
        Name associated with dimension, e.g., "number of trees".

    * `overlap` [float, default=0.25]:
        Amount of overlap between between each hyperspace.
        - Should be between 0 and 1.
        - If overlap=0, there are no shared values between the hyperspaces.
        - If overlap=1, two copies of the search space is made.
    """
    def __init__(self, low, high, prior="uniform", transform=None, overlap=0.25, name=None):
        super().__init__(low, high, prior, transform)
        self.prior = prior
        self.transform = transform
        self.overlap = overlap
        self.name = name
        self.space0_low = None
        self.space0_high = None
        self.space1_low = None
        self.space1_high = None
        self._divide_space()

        if high <= low:
            raise ValueError("the lower bound {} has to be less than the"
                             " upper bound {}".format(low, high))

    def __repr__(self):
        """
        Representation of the Integer HyperSpace. Useful when checking the hyperspace bounds.
        """
        return "HyperReal(low={}, high={}, prior={}, transform={})\n" \
               "HyperReal(low={}, high={}, prior={}, transform={})" \
               "".format(self.space0_low, self.space0_high, self.prior, self.transform,
                         self.space1_low, self.space1_high, self.prior, self.transform)

    def _divide_space(self):
        """
        Divides the original search space into overlapping subspaces.
        """
        subinterval_length = abs(self.high - self.low)/2
        overlap_length = subinterval_length * self.overlap

        if subinterval_length < 1:
            warnings.warn("Each hyperspace contains a single value.")

        self.space0_low = self.low
        self.space0_high = self.space0_low + subinterval_length + overlap_length
        self.space1_low = self.high - (subinterval_length + overlap_length)
        self.space1_high = self.high

    def get_hyperspace(self):
        """
        Create integer HyperSpaces.
        """
        return Real(self.space0_low, self.space0_high, self.prior, self.transform), \
               Real(self.space1_low, self.space1_high, self.prior, self.transform)


class HyperCategorical(HyperSpace, Categorical):
    """Search space dimension that can take on categorical values.

    Parameters
    ----------
    * `categories` [list, shape=(n_categories,)]:
        Sequence of possible categories.

    * `prior` [list, shape=(categories,), default=None]:
        Prior probabilities for each category. By default all categories
        are equally likely.

    * `transform` ["onehot", "identity", default="onehot"] :
        - "identity", the transformed space is the same as the original
          space.
        - "onehot", the transformed space is a one-hot encoded
          representation of the original space.

    * `name` [str or None]:
        Name associated with dimension, e.g., "colors".
    """
    def __init__(self, categories, prior=None, transform=None, overlap=0.25, name=None):
        super().__init__(categories, prior, transform)
        self.categories = categories
        self.prior = prior
        self.transform = transform
        self.overlap = overlap
        self.name = name
        self.cat_low = None
        self.cat_high = None
        self._divide_space()

    def __repr__(self):
        if len(self.categories) > 7:
            cats = self.categories[:3] + [_Ellipsis()] + self.categories[-3:]
        else:
            cats = self.categories

        if self.prior is not None and len(self.prior) > 7:
            prior = self.prior[:3] + [_Ellipsis()] + self.prior[-3:]
        else:
            prior = self.prior

        return "Categorical(categories={}, prior={})".format(cats, prior)

    def _divide_space(self):
        """
        Divides the original search space into overlapping subspaces.
        """
        subinterval_length = floor(len(self.categories)/2)
        overlap_length = ceil(subinterval_length * self.overlap)

        if subinterval_length < 1:
            warnings.warn("Each hyperspace contains a single value.")

        cat_low = self.categories[0:subinterval_length + overlap_length]
        self.cat_low = tuple(cat_low)
        cat_reverse = self.categories[::-1]
        cat_high = cat_reverse[0:subinterval_length + overlap_length]
        self.cat_high = tuple(cat_high[::-1])

    def get_hyperspace(self):
        """
        Create integer HyperSpaces.
        """
        return Categorical(self.cat_low, self.prior, self.transform), \
               Categorical(self.cat_high, self.prior, self.transform)
