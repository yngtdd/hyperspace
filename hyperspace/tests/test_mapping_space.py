import pytest
from sklearn.utils.testing import assert_equal

from skopt.space import Space

from hyperspace.space import HyperInteger
from hyperspace.space import HyperReal
from hyperspace.space import HyperCategorical

from hyperspace.space.skopt.mapping_space import check_dimension
from hyperspace.space.skopt.mapping_space import fold_spaces
from hyperspace.space.skopt.mapping_space import create_hyperspace


@pytest.mark.fast_test
def check_int(low=0, high=10):
    """
    Test check_dimension with Integer values.
    """
    check_low, check_high = check_dimension((low, high))
    hyper_int = HyperInteger(low=low, high=high)
    hyper_low, hyper_high = hyper_int.get_hyperspace()

    assert_equal(check_low, hyper_low)
    assert_equal(check_high, hyper_high)


@pytest.mark.fast_test
def check_real(low=0, high=10.0):
    """
    Test check_dimension with Real values.
    """
    check_low, check_high = check_dimension((low, high))
    hyper_real = HyperReal(low=low, high=high)
    hyper_low, hyper_high = hyper_real.get_hyperspace()

    assert_equal(check_low, hyper_low)
    assert_equal(check_high, hyper_high)


@pytest.mark.fast_test
def check_categorical(categories=['a', 'b', 'c', 'd']):
    """
    Test check_dimension with categorical values.
    """
    check_low, check_high = check_dimension(categories)
    hyper_cat = HyperCategorical(categories)
    hyper_low, hyper_high = hyper_cat.get_hyperspace()

    assert_equal(check_low, hyper_low)
    assert_equal(check_high, hyper_high)


@pytest.mark.fast_test
def test_fold_spaces(integer=(0, 10), real=(20.0, 30), cat=['a', 'b', 'c', 'd']):
    """
    Test all combinations of hyperparameter spaces.
    """
    hparams = [integer, real, cat]
    hparams_low = []
    hparams_high = []
    for hparam in hparams:
        low, high = check_dimension(hparam)
        hparams_low.append(low)
        hparams_high.append(high)

    all_spaces = fold_spaces(hparams_low, hparams_high)

    # Integer hparam
    hyper_int = HyperInteger(low=integer[0], high=integer[1])
    int_low, int_high = hyper_int.get_hyperspace()

    # Real hparam
    hyper_real = HyperReal(low=real[0], high=real[1])
    real_low, real_high = hyper_real.get_hyperspace()

    # Categorical hparam
    hyper_cat = HyperCategorical(cat)
    cat_low, cat_high = hyper_cat.get_hyperspace()

    # All combinations:
    test_combinations = [[int_high, real_high, cat_high],
                         [int_low, real_high, cat_high],
                         [int_high, real_low, cat_high],
                         [int_low, real_low, cat_high],
                         [int_high, real_high, cat_low],
                         [int_low, real_high, cat_low],
                         [int_high, real_low, cat_low],
                         [int_low, real_low, cat_low]]

    hyperspace = []
    for space in all_spaces:
        hyperspace.append(Space(space))

    test_case = []
    for space in test_combinations:
        test_case.append(Space(space))

    assert_equal(hyperspace, test_case)


@pytest.mark.fast_test
def test_create_hyperspace(integer=(0, 10), real=(20.0, 30), cat=['a', 'b', 'c', 'd']):
    """
    Tests for correctly formatted combinations of Scikit-Optimize spaces.
    """
    hparams0 = [integer, real]
    hparams1 = [real, cat]
    hparams2 = [integer, cat]

    # Integer hparam
    hyper_int = HyperInteger(low=integer[0], high=integer[1])
    int_low, int_high = hyper_int.get_hyperspace()

    # Real hparam
    hyper_real = HyperReal(low=real[0], high=real[1])
    real_low, real_high = hyper_real.get_hyperspace()

    # Categorical hparam
    hyper_cat = HyperCategorical(cat)
    cat_low, cat_high = hyper_cat.get_hyperspace()

    # Case 0
    hyperspace0 = create_hyperspace(hparams0)
    test_combinations0 = [[int_high, real_high],
                          [int_low, real_high],
                          [int_high, real_low],
                          [int_low, real_low]]

    test_hyperspace0 = []
    for space in test_combinations0:
        test_hyperspace0.append(Space(space))

    # Case 1: [real, cat]
    hyperspace1 = create_hyperspace(hparams1)
    test_combinations1 = [[real_high, cat_high],
                          [real_low, cat_high],
                          [real_high, cat_low],
                          [real_low, cat_low]]

    test_hyperspace1 = []
    for space in test_combinations1:
        test_hyperspace1.append(Space(space))


    # Case 3: [int, cat]
    hyperspace2 = create_hyperspace(hparams2)
    test_combinations2 = [[int_high, cat_high],
                          [int_low, cat_high],
                          [int_high, cat_low],
                          [int_low, cat_low]]

    test_hyperspace2 = []
    for space in test_combinations2:
        test_hyperspace2.append(Space(space))

    assert_equal(hyperspace0, test_hyperspace0)
    assert_equal(hyperspace1, test_hyperspace1)
    assert_equal(hyperspace2, test_hyperspace2)


if __name__=='__main__':
    check_int()
    check_real()
    check_categorical()
    test_fold_spaces()
    test_create_hyperspace()
