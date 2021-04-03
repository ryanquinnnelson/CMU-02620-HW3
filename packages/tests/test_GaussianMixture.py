import numpy as np
from packages.mixture import GaussianMixture as gm
import pytest


def test__get_random_assignments():
    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])
    K = 3
    actual = gm._get_random_assignments(X, K)

    # ensure each row adds up to 1.0
    expected_column_totals = np.array([[1.0],
                                       [1.0],
                                       [1.0],
                                       [1.0]])
    np.testing.assert_allclose(np.sum(actual, axis=1).reshape(-1, 1), expected_column_totals, atol=1e-16)
    assert actual.shape == (4, 3)


def test__initialize_assignments__array_version():
    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])
    K = 3

    A_init = np.array([0.3, 0.3, 0.4])
    expected = np.array([[0.3, 0.3, 0.4],
                         [0.3, 0.3, 0.4],
                         [0.3, 0.3, 0.4],
                         [0.3, 0.3, 0.4]])
    actual = gm._use_initial_assignments(X, K, A_init)
    np.testing.assert_array_equal(actual, expected)


def test__initialize_assignments__matrix_version():
    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])
    K = 3

    A_init = np.array([[0.3, 0.1, 0.6],
                       [0.3, 0.5, 0.2],
                       [0.3, 0.3, 0.4],
                       [0.1, 0.1, 0.8]])

    actual = gm._use_initial_assignments(X, K, A_init)
    np.testing.assert_array_equal(actual, A_init)


def test__initialize_assignments__error_version():
    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])
    K = 3

    A_init = np.array([[0.3, 0.1, 0.6],
                       [0.3, 0.5, 0.2],
                       [0.3, 0.3, 0.4]])

    with pytest.raises(ValueError):
        gm._use_initial_assignments(X, K, A_init)


def test__check_assignments__success():
    A = np.array([[0.3, 0.1, 0.6],
                  [0.3, 0.5, 0.2],
                  [0.3, 0.3, 0.4],
                  [0.1, 0.1, 0.8]])

    gm._check_assignments(A)


def test__check_assignments__failure():
    A = np.array([[0.3, 0.1, 0.1],
                  [0.3, 0.5, 0.1],
                  [0.3, 0.3, 0.4],
                  [0.1, 0.1, 0.8]])

    with pytest.raises(AssertionError):
        gm._check_assignments(A)


def test__initialize_assignments__no_initial_assignment():
    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])

    K = 3
    A_init = None

    actual = gm._initialize_assignments(X, K, A_init)
    assert actual.shape == (4, 3)


def test__initialize_assignments__initial_assignment():
    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])

    K = 3
    A_init = np.array([0.3, 0.3, 0.4])
    expected = np.array([[0.3, 0.3, 0.4],
                         [0.3, 0.3, 0.4],
                         [0.3, 0.3, 0.4],
                         [0.3, 0.3, 0.4]])

    actual = gm._initialize_assignments(X, K, A_init)
    np.testing.assert_array_equal(actual, expected)


def test__initialize_parameters__no_initial_values():
    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])

    A = np.array([[0.3, 0.3, 0.4],
                  [0.3, 0.3, 0.4],
                  [0.3, 0.3, 0.4],
                  [0.3, 0.3, 0.4]])
    pi_init = None
    mu_init = None
    sigma_init = None

    actual_pi, actual_mu, actual_sigma = gm._initialize_parameters(X, A, pi_init, mu_init, sigma_init)
    assert len(actual_pi) == 3
    assert len(actual_mu) == 3
    assert len(actual_sigma) == 3


def test__initialize_parameters__all_initial_values():
    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])

    A = np.array([[0.3, 0.3, 0.4],
                  [0.3, 0.3, 0.4],
                  [0.3, 0.3, 0.4],
                  [0.3, 0.3, 0.4]])
    pi_init = [0.5, 1.0, 3.0]
    mu_init = [0.5, 1.0, 3.0]
    sigma_init = [0.5, 1.0, 3.0]

    actual_pi, actual_mu, actual_sigma = gm._initialize_parameters(X, A, pi_init, mu_init, sigma_init)
    assert actual_pi == pi_init
    assert actual_mu == mu_init
    assert actual_sigma == sigma_init


def test___init__no_parameters():
    K = 3
    epsilon = 1e-3

    model = gm.GaussianMixture(K, epsilon)

    assert model.K == K
    assert model.epsilon == epsilon
    assert model.pi_init is None
    assert model.mu_init is None
    assert model.sigma_init is None
    assert model.A_init is None
    assert model.pi is None
    assert model.mu is None
    assert model.sigma is None


def test__init__all_parameters():
    K = 3
    epsilon = 1e-3
    pi_init = [1, 2, 3]
    mu_init = [4, 5, 6]
    sigma_init = [7, 8, 9]
    A_init = [10, 11, 12]

    model = gm.GaussianMixture(K, epsilon, pi_init, mu_init, sigma_init, A_init)

    assert model.K == K
    assert model.epsilon == epsilon
    assert model.pi_init == pi_init
    assert model.mu_init == mu_init
    assert model.sigma_init == sigma_init
    assert model.A_init == A_init
    assert model.pi is None
    assert model.mu is None
    assert model.sigma is None


def test_fit():
    K = 3
    epsilon = 1e-1

    model = gm.GaussianMixture(K, epsilon)

    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])

    model.fit(X)
    assert len(model.pi) == 3
    assert len(model.mu) == 3
    assert len(model.sigma) == 3


def test_fit_and_score():
    K = 3
    epsilon = 1e-1

    model = gm.GaussianMixture(K, epsilon)

    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])

    scores = model.fit_and_score(X)
    assert len(model.pi) == 3
    assert len(model.mu) == 3
    assert len(model.sigma) == 3
    assert len(scores) > 0


def test_predict_proba__multiple_samples():

    X = np.array([[6, 4],
                  [4, 1]])

    # pi
    all_pi = [0.25, 0.25, 0.5]

    # mu
    all_mu = [np.array([4.3, 2.6]),
              np.array([3.9, 1.7]),
              np.array([2.4, 1.85])]

    # sigma
    all_sigma = [np.array([[3.61, 2.22],
                           [2.22, 2.04]]),
                 np.array([[2.09, 0.97],
                           [0.97, 1.41]]),
                 np.array([[3.04, 1.06],
                           [1.06, 1.0275]])]

    expected = np.array([[1,1,1],
                         [1,1,1]])

    model = gm.GaussianMixture(K=3, epsilon=1e-3)
    model.pi = all_pi
    model.mu = all_mu
    model.sigma = all_sigma

    actual = model.predict_proba(X)
    assert actual.shape == (2,3)