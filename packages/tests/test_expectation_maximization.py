from packages.mixture import expectation_maximization as em
import numpy as np


def test__sum_soft_assignments():
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.3, 0.4],
                  [0.1, 0.1, 0.8],
                  [0.2, 0.1, 0.7]])

    expected = np.array([1.1, 0.7, 2.2])
    actual = em._sum_soft_assignments(A)
    np.testing.assert_array_equal(actual, expected)


def test__sum_soft_assignments2():
    A = np.array([[0.5, 0.2, 0.3],
                  [0.2, 0.6, 0.2],
                  [0.1, 0.1, 0.8],
                  [0.2, 0.1, 0.7]])
    expected = np.array([1.0, 1.0, 2.0])
    actual = em._sum_soft_assignments(A)
    np.testing.assert_array_equal(actual, expected)


def test__calculate_all_pi():
    A = np.array([[0.5, 0.2, 0.3],
                  [0.2, 0.6, 0.2],
                  [0.1, 0.1, 0.8],
                  [0.2, 0.1, 0.7]])
    S = np.array([1.0, 1.0, 2.0])

    expected = [0.25, 0.25, 0.5]
    actual = em._calculate_all_pi(A, S)
    assert actual == expected
    assert np.sum(actual) == 1.0


def test__calculate_mu_k():
    A = np.array([[0.5, 0.2, 0.3],
                  [0.2, 0.6, 0.2],
                  [0.1, 0.1, 0.8],
                  [0.2, 0.1, 0.7]])
    S = np.array([1.0, 1.0, 2.0])

    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])

    # define expected results
    expected_mu0 = np.array([4.3, 2.6])
    expected_mu1 = np.array([3.9, 1.7])
    expected_mu2 = np.array([2.4, 1.85])
    expected = [expected_mu0, expected_mu1, expected_mu2]

    # test for expected results
    K = 3
    for k in range(K):
        actual = em._calculate_mu_k(X, A, S, k)
        np.testing.assert_allclose(actual, expected[k], atol=1e-16)  # rounding issues


def test__calculate_X_corr():
    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])
    mu_k = np.array([4.3, 2.6])

    expected = np.array([4.85, 2.65, 11.25, 7.85]).reshape(-1, 1)
    actual = em._calculate_X_corr(X, mu_k)
    np.testing.assert_allclose(actual, expected, atol=1e-16)


def test__calculate_sigma_k_0():
    A = np.array([[0.5, 0.2, 0.3],
                  [0.2, 0.6, 0.2],
                  [0.1, 0.1, 0.8],
                  [0.2, 0.1, 0.7]])
    S = np.array([1.0, 1.0, 2.0])

    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])
    mu_k = np.array([4.3, 2.6])
    k = 0

    expected = 5.65
    actual = em._calculate_sigma_k(X, A, S, mu_k, k)
    np.testing.assert_allclose(actual, expected, atol=1e-16)


def test__calculate_sigma_k_1():
    A = np.array([[0.5, 0.2, 0.3],
                  [0.2, 0.6, 0.2],
                  [0.1, 0.1, 0.8],
                  [0.2, 0.1, 0.7]])
    S = np.array([1.0, 1.0, 2.0])

    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])
    mu_k = np.array([3.9, 1.7])
    k = 1

    expected = 3.5
    actual = em._calculate_sigma_k(X, A, S, mu_k, k)
    np.testing.assert_allclose(actual, expected, atol=1e-16)


def test__calculate_sigma_k_2():
    A = np.array([[0.5, 0.2, 0.3],
                  [0.2, 0.6, 0.2],
                  [0.1, 0.1, 0.8],
                  [0.2, 0.1, 0.7]])
    S = np.array([1.0, 1.0, 2.0])

    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])
    mu_k = np.array([2.4, 1.85])
    k = 2

    expected = 4.0675
    actual = em._calculate_sigma_k(X, A, S, mu_k, k)
    np.testing.assert_allclose(actual, expected, atol=1e-16)


def test_m_step():
    A = np.array([[0.5, 0.2, 0.3],
                  [0.2, 0.6, 0.2],
                  [0.1, 0.1, 0.8],
                  [0.2, 0.1, 0.7]])

    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])
    # pi
    expected_pi = [0.25, 0.25, 0.5]

    # mu
    expected_mu = [np.array([4.3, 2.6]),
                   np.array([3.9, 1.7]),
                   np.array([2.4, 1.85])]

    # sigma
    expected_sigma = [5.65, 3.5, 4.0675]

    # actual results
    actual_pi, actual_mu, actual_sigma = em.m_step(X, A)

    # confirm results match expectation
    for k in range(3):
        # pi
        actual_pi_k = actual_pi[k]
        expected_pi_k = expected_pi[k]
        assert actual_pi_k == expected_pi_k

        # mu
        actual_mu_k = actual_mu[k]
        expected_mu_k = expected_mu[k]
        np.testing.assert_allclose(actual_mu_k, expected_mu_k, atol=1e-16)

        # sigma
        actual_sigma_k = actual_sigma[k]
        expected_sigma_k = expected_sigma[k]
        np.testing.assert_allclose(actual_sigma_k, expected_sigma_k, atol=1e-16)


def test__calculate_prob_Xk():
    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])

    # pi
    all_pi = [0.25, 0.25, 0.5]

    # mu
    all_mu = [np.array([4.3, 2.6]),
              np.array([3.9, 1.7]),
              np.array([2.4, 1.85])]

    # sigma
    all_sigma = [5.65, 3.5, 4.0675]

    k = 0
    expected = np.array([0.0045847,
                         0.00557011,
                         0.00260219,
                         0.0035157])
    actual = em._calculate_prob_Xk(X, all_pi, all_mu, all_sigma, k)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test___calculate_prob_X():
    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])

    # pi
    all_pi = [0.25, 0.25, 0.5]

    # mu
    all_mu = [np.array([4.3, 2.6]),
              np.array([3.9, 1.7]),
              np.array([2.4, 1.85])]

    # sigma
    all_sigma = [5.65, 3.5, 4.0675]

    # calculate expected probabilities
    p_X = np.array([[0.0045847, 0.00557011, 0.00260219, 0.0035157],  # p(X|c=1)*p(c=1)
                    [0.0028437, 0.01058452, 0.00337547, 0.00632877],  # p(X|c=2)*p(c=2)
                    [0.00225323, 0.01306843, 0.01533292, 0.01755293]])  # p(X|c=3)*p(c=3)

    expected = np.sum(p_X, axis=0).reshape(-1, 1)
    actual = em._calculate_prob_X(X, all_pi, all_mu, all_sigma)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
