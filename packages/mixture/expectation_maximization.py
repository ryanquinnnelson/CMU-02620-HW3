"""
Implements the Expectation-Maximization algorithm.
"""
import numpy as np
from scipy.stats import multivariate_normal


# tested
def _sum_soft_assignments(A):
    """
    Sums the soft assignment values for all samples for each cluster.
    :param A: N x K matrix, where N is the number of samples and K is the number of clusters.
            Represents the soft assignments for each clusters for all samples.
            assignments[n][k] contains the soft-assignment of the nth sample for the kth cluster
    :return: K x 1 array, where [k] is the sum of all soft assignments for the kth cluster
    """
    return np.sum(A, axis=0)


# tested
def _calculate_all_pi(A, S):
    """
    Calculates pi parameter for all K clusters simultaneously.
    pi for cluster k is defined as:

    pi_k = p(c=k) = a / N

    where
    - a = SUM_n=1^N p(c^n = k | X^n)
    - N is the number of samples
    - k is the kth cluster
    - X^n is the nth sample

    :param A: N x K matrix, where N is the number of samples and K is the number of clusters.
            Represents the soft assignments for each clusters for all samples.
            assignments[n][k] contains the soft-assignment of the nth sample for the kth cluster
    :param S: K x 1 array, where [k] is the sum of all soft assignments for the kth cluster
    :return: K x 1 array, where [k] is the pi parameter for cluster k
    """
    N = len(A)  # number of samples
    all_pi = S / N
    return all_pi.tolist()


# tested
# ?? seems odd to get mu as a vector (not a scalar)
def _calculate_mu_k(X, A, S, k):
    """
    Calculates mu parameter for cluster k. Uses the following formula for mu_k:

    mu_k = a / b

    where
    - a = SUM_n=1^N X^n * p(c^n=k|X^n)
    - b = SUM_n=1^N p(c^n = k | X^n)
    - N is the number of samples
    - k is the kth cluster
    - X^n is the nth sample

    :param X: N x J matrix, where N is the number of samples and J is the number of features per sample
    :param A: N x K matrix, where K is the number of clusters
    :param S: K x 1 array, represents sum of soft assignments for each cluster
    :param k: kth cluster, represents cluster for which parameter is being calculated
    :return: J x 1 array
    """
    a_k = A[:, k].reshape(-1, 1)  # soft assignments for kth cluster
    s_k = S[k]  # sum of soft assignments for kth cluster
    return np.sum(X * a_k, axis=0) / s_k


# tested
def _calculate_X_corr(X, mu_k):
    """
    (X^n - mu_k)(X^n - mu_k)^T

    This implementation takes advantage of the fact that (X^n - mu_k)(X^n - mu_k)^T is the same as squaring each
    dimension of the row vector then adding all terms together:

    Example:
    Assume (X^n - mu_k) = | 1.7 | -0.3 |

    | 1.7 | -0.3 |  x  |  1.7 |  = (1.7 * 1.7) + (-0.3 * -0.3) = 2.98
                       | -0.3 |

    The function does this for all X^n in parallel.

    :param X:
    :param mu_k:
    :return:
    """
    X_corr = X - mu_k
    return np.sum(np.square(X_corr), axis=1).reshape(-1, 1)


# tested
# ?? seems odd to get sigma as a scalar (not a matrix)
def _calculate_sigma_k(X, A, S, mu_k, k):
    """
    Calculates Sigma parameter for cluster k. Uses the following formula for sigma_k:

    sigma_k = a / b

    where
    - a = SUM_n=1^N (X^n - mu_k)(X^n - mu_k)^T p(c^n=k|X^n)
    - b = SUM_n=1^N p(c^n = k | X^n)
    - N is the number of samples
    - mu is the mu parameter for cluster k
    - k is the kth cluster
    - X^n is the nth sample

    :param X: N x J matrix, where N is the number of samples and J is the number of features per sample
    :param A: N x K matrix, where K is the number of clusters
    :param S: K x 1 array, represents sum of soft assignments for each cluster
    :param mu_k: J x 1 array, represents mu parameter for cluster k
    :param k: kth cluster, represents cluster for which parameter is being calculated
    :return: N x N matrix
    """
    X_corr = _calculate_X_corr(X, mu_k)
    a_k = A[:, k].reshape(-1, 1)
    s_k = S[k]

    sigma_k = np.sum(X_corr * a_k, axis=0) / s_k

    return sigma_k.item(0)


# tested
def m_step(X, A):
    """
    Performs MLE for parameter estimation using data and soft assignments.

    :param X: N x J matrix, where N is the number of samples and J is the number of features per sample
    :param A: N x K matrix, where K is the number of clusters
    :return: (list, list, list) Tuple representing ( pi, mu, sigma) parameters for all K clusters.
    """
    S = _sum_soft_assignments(A)  # used multiple times so calculated separately
    K = len(S)

    all_pi = _calculate_all_pi(A, S)
    all_mu = []
    all_sigma = []

    for k in range(K):
        mu_k = _calculate_mu_k(X, A, S, k)
        all_mu.append(mu_k)

        sigma_k = _calculate_sigma_k(X, A, S, mu_k, k)
        all_sigma.append(sigma_k)

    return all_pi, all_mu, all_sigma


# tested
def _calculate_prob_Xk(X, all_pi, all_mu, all_sigma, k):
    mu_k = all_mu[k]
    sigma_k = all_sigma[k]
    pi_k = all_pi[k]

    y = multivariate_normal.pdf(X, mean=mu_k, cov=sigma_k)
    return y * pi_k


def _calculate_prob_X(X, all_pi, all_mu, all_sigma):
    """
    Calculates P(X), the denominator in the inference equation. Uses the following formula:

    P(X^n) = SUM_k=1^K p(X^n|c^n=k) * p(c^n=k)

    :param X:
    :param all_pi:
    :param all_mu:
    :param all_sigma:
    :return:
    """
    N = len(X)
    K = len(all_pi)
    all_prob = np.zeros((K, N))

    for k in range(K):
        prob_Xk = _calculate_prob_Xk(X, all_pi, all_mu, all_sigma, k)
        all_prob[k] = prob_Xk

    return np.sum(all_prob, axis=0).reshape(-1, 1)

#
# def e_step(X, all_pi, all_mu, all_sigma):
#     """
#     Infers soft assignments to each cluster for each sample.
#
#     :param X:
#     :param all_pi:
#     :param all_mu:
#     :param all_sigma:
#     :return:
#     """
#     p_X = _calculate_prob_X(X, all_pi, all_mu, all_sigma)
#
#     N = len(X)
#     K = len(all_pi)
#     A = np.zeros((N, K))
#
#     for k in range(K):
#         p_Xk = _calculate_prob_Xk(X, all_pi, all_mu, all_sigma, k)
#         a_k = p_Xk / p_X
#         A[k] = a_k
#
#     return A
#
#
# def _calculate_log_likelihood(X, all_pi, all_mu, all_sigma):
#     p_X = _calculate_prob_X(X, all_pi, all_mu, all_sigma)
#     log_p_X = np.log(p_X)
#     return np.sum(log_p_X)
