"""
Implementation of Gaussian Mixture model.
"""
import numpy as np
from scipy.stats import multivariate_normal


def _sum_soft_assignments(A):
    """
    Sums the soft assignment values for all samples for each cluster.
    :param A: N x K matrix, where N is the number of samples and K is the number of clusters.
            Represents the soft assignments for each clusters for all samples.
            assignments[n][k] contains the soft-assignment of the nth sample for the kth cluster
    :return: K x 1 array, where [k] is the sum of all soft assignments for the kth cluster
    """
    return np.sum(A)


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
    return S / N


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
    a_k = A[:, k]  # soft assignments for kth cluster
    s_k = S[k]  # sum of soft assignments for kth cluster
    return X * a_k / s_k


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
    d_k = X - mu_k
    a_k = A[:, k]
    s_k = S[k]

    return d_k * d_k.T * a_k / s_k


def _m_step(X, A):
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

    return all_pi.tolist(), all_mu, all_sigma


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
        mu_k = all_mu[k]
        sigma_k = all_sigma[k]
        pi_k = all_pi[k]

        y = multivariate_normal.pdf(X, mean=mu_k, cov=sigma_k)
        prob_Xk = y * pi_k
        all_prob[k] = prob_Xk

    return np.sum(all_prob).reshape(-1, 1)


def _calculate_prob_Xk(X, all_pi, all_mu, all_sigma, k):
    mu_k = all_mu[k]
    sigma_k = all_sigma[k]
    pi_k = all_pi[k]

    y = multivariate_normal.pdf(X, mean=mu_k, cov=sigma_k)
    return y * pi_k


def _e_step(X, all_pi, all_mu, all_sigma):
    """
    Infers soft assignments to each cluster for each sample.

    :param X:
    :param all_pi:
    :param all_mu:
    :param all_sigma:
    :return:
    """
    p_X = _calculate_prob_X(X, all_pi, all_mu, all_sigma)

    N = len(X)
    K = len(all_pi)
    A = np.zeros((N, K))

    for k in range(K):
        p_Xk = _calculate_prob_Xk(X, all_pi, all_mu, all_sigma, k)
        a_k = p_Xk / p_X
        A[k] = a_k

    return A


def _initialize_assignments(X, K):
    N = len(X)
    A_T = np.zero((K, N))

    for k in range(K):
        # generate random probabilities for each sample
        p_ck = np.random.uniform(size=N)
        A_T[k] = p_ck

    return A_T.T


def _predicted_cluster_changed(y_prev, y_curr):
    cluster_changed = np.sum(y_prev - y_curr)
    return cluster_changed == 0.0


def _predict_clusters(A):
    # get index (cluster) of highly probability assignment for each sample
    predictions = np.argmax(A, axis=1)
    return predictions


def _calculate_log_likelihood(p_X):
    log_p_X = np.log(p_X)
    return np.sum(log_p_X)


class GaussianMixture:

    def __init__(self, k):
        self.k = k

    def fit(self, X):

        A = _initialize_assignments(X, self.k)
        y_prev = _predict_clusters(A)

        prediction_changed = True
        counter = 0
        while prediction_changed:

            # m step
            all_pi, all_mu, all_sigma = _m_step(X, A)

            # e step
            A = _e_step(X, all_pi, all_mu, all_sigma)

            # predict clusters
            y_curr = _predict_clusters(A)

            # check for convergence
            prediction_changed = _predicted_cluster_changed(y_prev, y_curr)
            if prediction_changed:
                y_prev = y_curr  # save for next iteration
                counter += 1

            if counter > 100000:
                break  # something is probably wrong

        return self

    def fit_and_score(self, X):

        A = _initialize_assignments(X, self.k)
        y_prev = _predict_clusters(A)

        prediction_changed = True
        counter = 0
        log_likelihoods = []
        while prediction_changed:

            # m step
            all_pi, all_mu, all_sigma = _m_step(X, A)

            # e step
            A = _e_step(X, all_pi, all_mu, all_sigma)

            # predict clusters
            y_curr = _predict_clusters(A)

            # check for convergence
            prediction_changed = _predicted_cluster_changed(y_prev, y_curr)
            if prediction_changed:
                y_prev = y_curr  # save for next iteration
                counter += 1

            if counter > 100000:
                break  # something is probably wrong

            # calculate score
            p_X = _calculate_prob_X(X, all_pi, all_mu, all_sigma)
            log_likelihood = _calculate_log_likelihood(p_X)
            log_likelihoods.append(log_likelihood)

        return log_likelihoods
