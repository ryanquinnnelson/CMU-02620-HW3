"""
Implementation of Gaussian Mixture model.
"""
import numpy as np
from packages.mixture import expectation_maximization as em


def _initialize_assignments(X, K):
    N = len(X)
    A_T = np.zeros((K, N))

    for k in range(K):
        # generate random probabilities for each sample
        p_ck = np.random.uniform(size=N)
        A_T[k] = p_ck

    return A_T.T


def _assignments_changed(y_prev, y_curr):
    cluster_changed = np.sum(y_prev - y_curr, axis=0)
    return cluster_changed == 0.0


def _assign_clusters(A):
    # get index (cluster) of highly probability assignment for each sample
    predictions = np.argmax(A, axis=1)
    return predictions


class GaussianMixture:

    def __init__(self, k):
        self.k = k

    def fit(self, X):

        A = _initialize_assignments(X, self.k)
        y_prev = _assign_clusters(A)

        assignments_changed = True
        counter = 0
        while assignments_changed:

            # m step
            all_pi, all_mu, all_sigma = em.m_step(X, A)

            # e step
            A = em.e_step(X, all_pi, all_mu, all_sigma)

            # assign clusters
            y_curr = _assign_clusters(A)

            # check for convergence
            assignments_changed = _assignments_changed(y_prev, y_curr)

            # cleanup steps
            if assignments_changed:
                y_prev = y_curr  # save for next iteration
                counter += 1

            if counter > 100000:
                break  # something is probably wrong

        return self

    def fit_and_score(self, X):

        A = _initialize_assignments(X, self.k)
        y_prev = _assign_clusters(A)

        prediction_changed = True
        counter = 0
        log_likelihoods = []
        while prediction_changed:

            # m step
            all_pi, all_mu, all_sigma = em.m_step(X, A)

            # e step
            A = em.e_step(X, all_pi, all_mu, all_sigma)

            # predict clusters
            y_curr = _assign_clusters(A)

            # check for convergence
            prediction_changed = _assignments_changed(y_prev, y_curr)
            if prediction_changed:
                y_prev = y_curr  # save for next iteration
                counter += 1

            if counter > 100000:
                break  # something is probably wrong

            # calculate score
            log_likelihood = em.calculate_log_likelihood(X, all_pi, all_mu, all_sigma)
            log_likelihoods.append(log_likelihood)

        return log_likelihoods
