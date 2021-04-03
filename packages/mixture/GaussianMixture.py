"""
Implementation of Gaussian Mixture model.
"""
import numpy as np
from packages.mixture import expectation_maximization as em


# tested
def _get_random_assignments(X, K):
    """
    Sets initial soft-assignments for all samples for all clusters using random values drawn from a uniform
    distribution. Standardizes soft-assignments for each cluster so assignments add up to 1.0 for a given sample.

    :param X: N x J matrix, where N is the number of samples and J is the number of features per sample.
    :param K: int, number of clusters in model.
    :return: N x K matrix, where N is the number of samples and K is the number of clusters.
            Represents the soft assignments for each cluster for all samples.
            A[n][k] is the soft-assignment of the nth sample for the kth cluster.
    """
    N = len(X)
    A_T = np.zeros((K, N))  # flipped clusters and samples to make it easy to replace rows

    for k in range(K):  # generate random probabilities for all clusters
        p_ck = np.random.uniform(size=N)
        A_T[k] = p_ck

    # scale values derived so every column adds up to 1.0
    sum_A_T = np.sum(A_T, axis=0)
    A_T_scaled = A_T / sum_A_T
    A = A_T_scaled.T
    return A


# tested
def _use_initial_assignments(X, K, A_init):
    """
    Sets initial soft-assignments for all samples for all clusters using given assignments.
    Raises ValueError if A_init is not K x 1 or N x K.

    :param X: N x J matrix, where N is the number of samples and J is the number of features per sample.
    :param K: int, number of clusters in model.
    :param A_init: K x 1 array or N X K matrix of initial soft assignments.
    :return: N x K matrix, where N is the number of samples and K is the number of clusters.
             Represents the soft assignments for each cluster for all samples.
             A[n][k] is the soft-assignment of the nth sample for the kth cluster.
    """
    N = len(X)
    if A_init.shape == (K,):  # A is an 1 x K array, assign same values to every sample

        A = np.zeros((N, K))
        A[:] = A_init

    elif A_init.shape == (N, K):
        A = A_init  # A is an N x K matrix, use it directly
    else:
        raise ValueError('Initial soft assignments matrix must be provided as a K x 1 array or N x K matrix.', A_init)

    return A


# tested
def _check_assignments(A):
    """
    Confirms each row of soft assignments adds up to 1. Raises AssertionError if any row does not.

    :param A: N x K matrix, where N is the number of samples and K is the number of clusters.
             Represents the soft assignments for each cluster for all samples.
             A[n][k] is the soft-assignment of the nth sample for the kth cluster.
    :return: None
    """
    expected = np.ones((len(A),))
    np.testing.assert_allclose(np.sum(A, axis=1), expected, atol=1e-16)


# tested
def _initialize_assignments(X, K, A):
    """
    Sets initial soft-assignments for all samples for all clusters.
    Ensures the soft-assignments for each cluster add up to 1.0 for a given sample.

    :param X: N x J matrix, where N is the number of samples and J is the number of features per sample.
    :param K: int, number of clusters in model.
    :return: N x K matrix, where N is the number of samples and K is the number of clusters.
            Represents the soft assignments for each cluster for all samples.
            A[n][k] is the soft-assignment of the nth sample for the kth cluster.
    """

    if A is None:  # an initial assignment is not provided
        A = _get_random_assignments(X, K)
    else:
        A = _use_initial_assignments(X, K, A)

    _check_assignments(A)
    return A


# def _assign_clusters(A):
#     # get index (cluster) of highly probability assignment for each sample
#     predictions = np.argmax(A, axis=1)
#     return predictions


# tested
def _initialize_parameters(X, A, pi_init, mu_init, sigma_init):
    # calculate initial values based on A
    all_pi, all_mu, all_sigma = em.m_step(X, A)

    if pi_init is not None:
        all_pi = pi_init

    if mu_init is not None:
        all_mu = mu_init

    if sigma_init is not None:
        all_sigma = sigma_init

    return all_pi, all_mu, all_sigma


class GaussianMixture:

    def __init__(self, K, epsilon, pi_init=None, mu_init=None, sigma_init=None, A_init=None):
        """
        Initializes Gaussian Mixture model with hyperparameters and also allows initial parameters to be passed in.

        :param K: int, number of clusters in model.
        :param epsilon: float, difference threshold for convergence calculation. Convergence is defined as the point
                        at which the difference in the objective function between iterations is less than epsilon.
        :param pi_init:
        :param mu_init:
        :param sigma_init:
        :param A_init:
        """
        self.K = K
        self.epsilon = epsilon
        self.pi_init = pi_init
        self.mu_init = mu_init
        self.sigma_init = sigma_init
        self.A_init = A_init
        self.pi = None
        self.mu = None
        self.sigma = None

    def fit_and_score(self, X):
        """

        :param X:
        :return:
        """

        # initialization
        scores = []
        A = _initialize_assignments(X, self.K, self.A_init)
        all_pi, all_mu, all_sigma = _initialize_parameters(X, A, self.pi_init, self.mu_init, self.sigma_init)

        # calculate first objective value
        prev_objective = em.calculate_log_likelihood(X, all_pi, all_mu, all_sigma)
        scores.append(prev_objective)

        counter = 0
        diff = self.epsilon  # to enter the while loop
        while diff >= self.epsilon:

            # e step - update assignments
            A = em.e_step(X, all_pi, all_mu, all_sigma)

            # m step - estimate parameters
            all_pi, all_mu, all_sigma = em.m_step(X, A)

            # calculate difference between previous and current objective values
            curr_objective = em.calculate_log_likelihood(X, all_pi, all_mu, all_sigma)
            scores.append(curr_objective)

            # take actions depending on whether there will be another round
            diff = np.abs(curr_objective - prev_objective)
            if diff >= self.epsilon:  # there will be another iteration
                prev_objective = curr_objective  # save for next round
                counter += 1

            # determine if there is an infinite loop
            if counter > 10000:
                raise ValueError('Model took too many iterations to converge:', counter)  # something is wrong

        # model converged, save parameters
        self.pi = all_pi
        self.mu = all_mu
        self.sigma = all_sigma

        print(counter)  # for fun, number of iterations it took to converge
        return scores

    def fit(self, X):
        """

        :param X:
        :return:
        """
        self.fit_and_score(X)
        return self
