import pytest
from sklearn.utils.testing import assert_equal

from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from hyperspace.space import HyperInteger
from hyperspace.space import HyperReal
from hyperspace.space import HyperCategorical

from math import floor, ceil


@pytest.mark.fast_test
def test_hyperinteger(low=0, high=10, overlap=0.25):
    """
    Tests that HyperInteger returns correctly formatted Scikit-Optimize Integers.
    """
    hyperinteger = HyperInteger(low=low, high=high, overlap=overlap)
    hyperspace_low, hyperspace_high = hyperinteger.get_hyperspace()

    subinterval_length = abs(high - low)/2
    overlap_length = subinterval_length * overlap

    space0_upperbound = floor(low + subinterval_length + overlap_length)
    space_low = Integer(low, space0_upperbound)

    space1_lowerbound = ceil(high - (subinterval_length + overlap_length))
    space_high = Integer(space1_lowerbound, high)

    assert_equal(hyperspace_low, space_low)
    assert_equal(hyperspace_high, space_high)


@pytest.mark.fast_test
def test_hyperreal(low=0, high=10.0, overlap=0.25):
    """
    Tests that HyperReal returns correctly formatted Scikit-Optimize Reals.
    """
    hyperreal = HyperReal(low=low, high=high, overlap=overlap)
    hyperspace_low, hyperspace_high = hyperreal.get_hyperspace()

    subinterval_length = abs(high - low)/2
    overlap_length = subinterval_length * overlap

    space0_upperbound = round(low + subinterval_length + overlap_length)
    space_low = Real(low, space0_upperbound)

    space1_lowerbound = round(high - (subinterval_length + overlap_length))
    space_high = Real(space1_lowerbound, high)

    assert_equal(hyperspace_low, space_low)
    assert_equal(hyperspace_high, space_high)


@pytest.mark.fast_test
def test_hypercat(categories=['a', 'b', 'c', 'd'], overlap=0.25):
    """
    Tests that HyperCategorical returns correctly formatted Scikit-Optimize Categories.
    """
    hypercat = HyperCategorical(categories=categories)
    hyperspace_low, hyperspace_high = hypercat.get_hyperspace()

    subinterval_length = floor(len(categories)/2)
    overlap_length = ceil(subinterval_length * overlap)

    cat_low = categories[0:subinterval_length + overlap_length]
    cat_reverse = categories[::-1]
    cat_high = cat_reverse[0:subinterval_length + overlap_length]
    cat_high = cat_high[::-1]

    space_low = Categorical(cat_low)
    space_high = Categorical(cat_high)

    assert_equal(hyperspace_low, space_low)
    assert_equal(hyperspace_high, space_high)


if __name__=='__main__':
    test_hyperinteger()
    test_hyperreal()
    test_hypercat()
