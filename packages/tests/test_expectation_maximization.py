from packages.mixture import expectation_maximization as em
import numpy as np
import pytest


def test__sum_soft_assignments():
    A = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.3, 0.4],
                  [0.1, 0.1, 0.8],
                  [0.2, 0.1, 0.7]])

    expected = np.array([1.1, 0.7, 2.2])
    actual = em._sum_soft_assignments(A)
    np.testing.assert_array_equal(actual, expected)


def test__sum_soft_assignments__2():
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


def test__calculate_sigma_k2__0():
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

    expected = np.array([[3.61, 2.22],
                         [2.22, 2.04]])
    actual = em._calculate_sigma_k2(X, A, S, mu_k, k)
    np.testing.assert_allclose(actual, expected, atol=1e-16)


def test__calculate_sigma_k2__1():
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

    expected = np.array([[2.09, 0.97],
                         [0.97, 1.41]])
    actual = em._calculate_sigma_k2(X, A, S, mu_k, k)
    np.testing.assert_allclose(actual, expected, atol=1e-16)


def test__calculate_sigma_k2__2():
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

    expected = np.array([[3.04, 1.06],
                         [1.06, 1.0275]])
    actual = em._calculate_sigma_k2(X, A, S, mu_k, k)
    np.testing.assert_allclose(actual, expected, atol=1e-16)

def test__calculate_sigma_k__0():
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

    expected = np.array([[3.61, 2.22],
                         [2.22, 2.04]])
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
    expected_sigma = [np.array([[3.61, 2.22],
                                [2.22, 2.04]]),
                      np.array([[2.09, 0.97],
                                [0.97, 1.41]]),
                      np.array([[3.04, 1.06],
                                [1.06, 1.0275]])]

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
    all_sigma = [np.array([[3.61, 2.22],
                           [2.22, 2.04]]),
                 np.array([[2.09, 0.97],
                           [0.97, 1.41]]),
                 np.array([[3.04, 1.06],
                           [1.06, 1.0275]])]

    k = 0
    expected = np.array([0.0155642, 0.00570463, 0.0012414, 0.01194359])
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
    all_sigma = [np.array([[3.61, 2.22],
                           [2.22, 2.04]]),
                 np.array([[2.09, 0.97],
                           [0.97, 1.41]]),
                 np.array([[3.04, 1.06],
                           [1.06, 1.0275]])]

    # calculate expected probabilities for each cluster
    all_prob = np.array([[0.0155642, 0.00570463, 0.0012414, 0.01194359],
                         [0.00391738, 0.02096575, 0.00091602, 0.01164242],
                         [0.00363336, 0.0081879, 0.02991268, 0.03734345]])

    expected = np.sum(all_prob, axis=0).reshape(-1, 1)
    actual = em._calculate_prob_X(X, all_pi, all_mu, all_sigma)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_calculate_log_likelihood():
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
    all_sigma = [np.array([[3.61, 2.22],
                           [2.22, 2.04]]),
                 np.array([[2.09, 0.97],
                           [0.97, 1.41]]),
                 np.array([[3.04, 1.06],
                           [1.06, 1.0275]])]

    # calculate expected probabilities for each cluster
    all_prob = np.array([[0.0155642, 0.00570463, 0.0012414, 0.01194359],
                         [0.00391738, 0.02096575, 0.00091602, 0.01164242],
                         [0.00363336, 0.0081879, 0.02991268, 0.03734345]])

    p_X = np.sum(all_prob, axis=0).reshape(-1, 1)
    log_p_X = np.log(p_X)

    expected = np.sum(log_p_X, axis=0).item(0)
    actual = em.calculate_log_likelihood(X, all_pi, all_mu, all_sigma)
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_e_step():
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
    all_sigma = [np.array([[3.61, 2.22],
                           [2.22, 2.04]]),
                 np.array([[2.09, 0.97],
                           [0.97, 1.41]]),
                 np.array([[3.04, 1.06],
                           [1.06, 1.0275]])]

    # calculate expected probabilities for each cluster
    all_prob = np.array([[0.0155642, 0.00570463, 0.0012414, 0.01194359],
                         [0.00391738, 0.02096575, 0.00091602, 0.01164242],
                         [0.00363336, 0.0081879, 0.02991268, 0.03734345]])

    p_X = np.sum(all_prob, axis=0).reshape(-1, 1)

    expected_k0 = all_prob[0].reshape(-1, 1) / p_X
    expected_k1 = all_prob[1].reshape(-1, 1) / p_X
    expected_k2 = all_prob[2].reshape(-1, 1) / p_X
    expected = np.concatenate([expected_k0, expected_k1, expected_k2], axis=1)

    actual = em.e_step(X, all_pi, all_mu, all_sigma)
    np.testing.assert_allclose(actual, expected, atol=1e-7)

    # ensure each row sums to 1.0
    expected_column_totals = np.array([[1.0],
                                       [1.0],
                                       [1.0],
                                       [1.0]])
    np.testing.assert_allclose(np.sum(actual, axis=1).reshape(-1, 1), expected_column_totals, atol=1e-16)


# def test__check_valid_assignment__success():
#     A = np.array([[0.36007, 0.00284, 0.63709],
#                   [0.17928, 0.58114, 0.23958],
#                   [0.15136, 0.38740, 0.46124],
#                   [0.20585, 0.41395, 0.38020]])
#
#     em._check_valid_assignment(A)
#
#
# def test__check_valid_assignment__failure():
#     A = np.array([[0.36006613, 0.1, 0.63709776],
#                   [0.17927669, 0.58113916, 0.23958415],
#                   [0.15136286, 0.38740025, 0.4612369],
#                   [0.20584788, 0.41394991, 0.38020221]])
#
#     with pytest.raises(AssertionError):
#         em._check_valid_assignment(A)
#
#
# def test__round_assignments():
#     A = np.array([[0.36006613, 0.00283611, 0.63709776],
#                   [0.17927669, 0.58113916, 0.23958415],
#                   [0.15136286, 0.38740025, 0.4612369],
#                   [0.20584788, 0.41394991, 0.38020221]])
#
#     expected = np.array([[0.36007, 0.00284, 0.63709],
#                   [0.17928, 0.58114, 0.23958],
#                   [0.15136, 0.38740, 0.46124],
#                   [0.20585, 0.41395, 0.38020]])
#     actual = em._round_assignments(A)
#     np.testing.assert_allclose(actual, expected, atol=1e-16)
