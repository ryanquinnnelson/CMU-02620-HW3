from packages.mixture import expectation_maximization as em
import numpy as np


def test__sum_soft_assignments():
    A = np.array([[0.5, 0.2, 0.3],
                  [0.4, 0.1, 0.5],
                  [0.1, 0.05, 0.85]])

    expected = np.array([0.9,0.35,1.65])
    actual = em._sum_soft_assignments(A)
    np.testing.assert_array_equal(actual, expected)
