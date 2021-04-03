"""
Implementation of Gaussian Mixture model.
"""
import numpy as np
from packages.mixture import expectation_maximization as em


# tested
def _initialize_assignments(X, K):
    """
    Sets initial soft-assignments for all samples for all clusters.
    Ensures the soft-assignments for each cluster add up to 1.0 for a given sample.

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

    return A_T_scaled.T


# def _assign_clusters(A):
#     # get index (cluster) of highly probability assignment for each sample
#     predictions = np.argmax(A, axis=1)
#     return predictions


class GaussianMixture:

    def __init__(self, K, epsilon):
        """

        :param K: int, number of clusters in model.
        :param epsilon: float, difference threshold for convergence calculation. Convergence is defined as the point
        at which the difference in the objective function between iterations is less than epsilon.
        """
        self.K = K
        self.epsilon = epsilon
        self.pi = None
        self.mu = None
        self.sigma = None

    def fit_and_score(self,X):
        scores = []

        A = _initialize_assignments(X, self.K)

        # m step - estimate parameters
        all_pi, all_mu, all_sigma = em.m_step(X, A)

        # calculate difference between previous and current objective values
        prev_objective = em.calculate_log_likelihood(X, all_pi, all_mu, all_sigma)
        scores.append(prev_objective)

        diff = self.epsilon  # to enter the while loop
        counter = 0
        while diff >= self.epsilon:

            # e step - update assignments
            A = em.e_step(X, all_pi, all_mu, all_sigma)

            # m step - estimate parameters
            all_pi, all_mu, all_sigma = em.m_step(X, A)

            # calculate difference between previous and current objective values
            curr_objective = em.calculate_log_likelihood(X, all_pi, all_mu, all_sigma)
            scores.append(curr_objective)
            diff = np.abs(curr_objective - prev_objective)

            # take actions depending on whether there will be another round
            if diff >= self.epsilon:  # there will be another iteration
                prev_objective = curr_objective  # save for next round
                counter += 1

            if counter > 10000:
                raise ValueError('Model took too many iterations to converge:', counter)  # something is wrong

        # model converged, save parameters
        self.pi = all_pi
        self.mu = all_mu
        self.sigma = all_sigma

        print(counter)  # for fun, see how many iterations it took to converge
        return scores

    def fit(self, X):
        self.fit_and_score(X)
        return self
