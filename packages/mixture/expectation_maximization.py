"""
Implements the Expectation-Maximization algorithm.
"""
import numpy as np
from scipy.stats import multivariate_normal

"""
Note 1 - Explanation of sigma_k calculation
The numerator in the formula for sigma_k is written as the sum of 
p(c^N = k |X^n) * (X^n - mu_k)(X^n - mu_k)^T for each n in N samples. 

Taken as written, this tells us to perform a matrix multiplication for each of N samples, then add the results together.
However, it is possible to perform a single matrix multiplication over X and achieve the same result.
This will be shown by arriving at the result via both strategies.

X =  [[6 4]     mu_k = [4.3 2.6]   (X-mu_k)^T = [[ 1.7  1.4]     p(c=1|X) = [[0.5]
      [4 1]                                      [-0.3 -1.6]                 [0.2]
      [1 2]                                      [-3.3 -0.6]                 [0.1]
      [2 1]]                                     [-2.3 -1.6]]                [0.2]]

Note that X1-mu_k results in a column vector. This ensures the multiplications produce a 2x2 matrix instead of a scalar.

X1-mu_k = [[6]   - [[4.3]   = [[1.7]
           [4]]     [2.6]]     [1.4]]

-----------------------------------------------------
Strategy 1 - separate multiplications for each sample
-----------------------------------------------------

(X1-mu_k)(X1-mu_k)^T * p(c1 = k | X1) = [[1.7]  x [[1.7 1.4]]     = [[2.89 2.38]   * 0.5  = [[1.445 1.19 ]
                                         [1.4]]                      [2.38 1.96]]            [1.19  0.98 ]]
                                         
(X2-mu_k)(X2-mu_k)^T * p(c2 = k | X2) = [[-0.3]  x [[-0.3 1.6]]   = [[0.09 0.48]   * 0.2  = [[0.018 0.096]
                                         [-1.6]]                     [0.48 2.56]]            [0.096 0.512]]

(X3-mu_k)(X3-mu_k)^T * p(c3 = k | X3) = [[-3.3]  x [[-3.3 -0.6]]  = [[10.89 1.98]  * 0.1  = [[1.089 0.198]
                                         [-0.6]]                     [1.98 0.36]]            [0.198 0.036]]             

(X4-mu_k)(X4-mu_k)^T * p(c4 = k | X4) = [[-2.3]  x [[-2.3 -1.6]]  = [[5.29 3.68]   * 0.2  = [[1.058 0.736]
                                         [-1.6]]                     [3.68 2.56]]            [0.736 0.512]]     

Sum of the four matrices:
[[3.61 2.22]
 [2.22 2.04]]
 
 
-----------------------------------------------------
Strategy 2 - single multiplication
-----------------------------------------------------
In this case, we can achieve the same result by multiplying p(c^n = k | X^n) * (X^n-mu_k) before performing a 
matrix multiplication with (X^n-mu_k)^T:


p(c^n = k | X^n) * (X^n-mu_k) = [[0.5]  * [[ 1.7  1.4]  = [[ 0.85  0.7 ]
                                 [0.1]     [-0.3 -1.6]     [-0.06 -0.32]
                                 [0.2]     [-3.3 -0.6]     [-0.33 -0.06]
                                 [0.1]]    [-2.3 -1.6]]    [-0.46 -0.32]]

Note that we only multiply (X^n-mu_k), not both (X^n-mu_k) and (X^n-mu_k)^T.
We transpose the result before multiplying by (X^n-mu_k)^T to achieve a 2 x 2 square matrix.

(p(c^n = k | X^n) * (X^n-mu_k)) (X^n-mu_k)^T = [[ 0.85 -0.06 -0.33 -0.46]   *   [[ 1.7  1.4]  =   [[3.61 2.22]
                                                [ 0.7  -0.32 -0.06 -0.32]]       [-0.3 -1.6]       [2.22 2.04]]
                                                                                 [-3.3 -0.6] 
                                                                                 [-2.3 -1.6]]
Sum of the four matrices:
[[3.61 2.22]
 [2.22 2.04]]
 
The two results are exactly the same. Strategy 2 requires fewer calculations, so that is the one that is used.
"""



# tested
def _sum_soft_assignments(A):
    """
    Sums the soft assignment values for all samples for each cluster.
    This step is performed separately to avoid repeating work when calculating pi, mu, and sigma parameters.

    :param A: N x K matrix, where N is the number of samples and K is the number of clusters.
            Represents the soft assignments for each cluster for all samples.
            A[n][k] is the soft-assignment of the nth sample for the kth cluster.
    :return: K x 1 array S, where S[k] is the sum of all soft assignments for the kth cluster.
    """
    return np.sum(A, axis=0)


# tested
def _calculate_all_pi(A, S):
    """
    Calculates pi parameter for all K clusters simultaneously. pi for cluster k is defined as:

    pi_k = p(c=k) = (1 / N) * SUM_n=1^N p(c^n = k | X^n)

    where
    - N is the number of samples
    - k is the kth cluster
    - X^n is the nth sample

    :param A: N x K matrix, where N is the number of samples and K is the number of clusters.
            Represents the soft assignments for each cluster for all samples.
            A[n][k] is the soft-assignment of the nth sample for the kth cluster.
    :param S: K x 1 array, where S[k] is the sum of all soft assignments for the kth cluster.
    :return: K x 1 list, where all_pi[k] is the pi parameter for the kth cluster.
    """
    # Clusters can be processed in parallel because this is simply summing each column and dividing by a scalar.
    N = len(A)
    all_pi = S / N
    return all_pi.tolist()


# tested
def _calculate_mu_k(X, A, S, k):
    """
    Calculates mu parameter for cluster k.

    mu_k = a / b

    where
    - a = SUM_n=1^N X^n * p(c^n=k|X^n)
    - b = SUM_n=1^N p(c^n = k | X^n)
    - N is the number of samples
    - k is the kth cluster
    - X^n is the nth sample

    :param X: N x J matrix, where N is the number of samples and J is the number of features per sample.
    :param A: N x K matrix, where N is the number of samples and K is the number of clusters.
            Represents the soft assignments for each cluster for all samples.
            A[n][k] is the soft-assignment of the nth sample for the kth cluster.
    :param S: K x 1 array, where S[k] is the sum of all soft assignments for the kth cluster.
    :param k: int, the kth cluster for which the parameter is being calculated. Zero-indexed.
    :return: J x 1 array, represents centroid of the kth cluster.
    """
    a_k = A[:, k].reshape(-1, 1)  # soft assignments for kth cluster
    s_k = S[k]  # sum of soft assignments for kth cluster
    mu_k = np.sum(X * a_k, axis=0) / s_k
    return mu_k


# tested
def _calculate_sigma_k(X, A, S, mu_k, k):
    """
    Calculates sigma parameter for cluster k.

    sigma_k = a / b

    where
    - a = SUM_n=1^N p(c^n=k|X^n)(X^n - mu_k)(X^n - mu_k)^T
    - b = SUM_n=1^N p(c^n = k | X^n)
    - N is the number of samples
    - mu_k is the mu parameter for cluster k
    - k is the kth cluster
    - X^n is the nth sample

    :param X: N x J matrix, where N is the number of samples and J is the number of features per sample.
    :param A: N x K matrix, where N is the number of samples and K is the number of clusters.
            Represents the soft assignments for each cluster for all samples.
            A[n][k] is the soft-assignment of the nth sample for the kth cluster.
    :param S: K x 1 array, where S[k] is the sum of all soft assignments for the kth cluster.
    :param mu_k: J x 1 array, represents centroid of the kth cluster.
    :param k: int, the kth cluster for which the parameter is being calculated. Zero-indexed.
    :return: N x N covariance matrix
    """
    X_minus_mu_k = X - mu_k
    a_k = A[:, k].reshape(-1, 1)  # soft assignments for kth cluster
    s_k = S[k]  # sum of soft assignments for kth cluster
    sigma_k = np.matmul((a_k * X_minus_mu_k).T, X_minus_mu_k) / s_k  # see Note 1 to explain calculation
    return sigma_k

#
# # tested
# def m_step(X, A):
#     """
#     Performs MLE for parameter estimation using data and soft assignments.
#
#     :param X: N x J matrix, where N is the number of samples and J is the number of features per sample
#     :param A: N x K matrix, where K is the number of clusters
#     :return: (list, list, list) Tuple representing ( pi, mu, sigma) parameters for all K clusters.
#     """
#     S = _sum_soft_assignments(A)  # used multiple times so calculated separately
#     K = len(S)
#
#     all_pi = _calculate_all_pi(A, S)
#     all_mu = []
#     all_sigma = []
#
#     for k in range(K):
#         mu_k = _calculate_mu_k(X, A, S, k)
#         all_mu.append(mu_k)
#
#         sigma_k = _calculate_sigma_k(X, A, S, mu_k, k)
#         all_sigma.append(sigma_k)
#
#     return all_pi, all_mu, all_sigma
#
#
# # tested
# # ?? Do I sum up over all X^n here?
# def _calculate_prob_Xk(X, all_pi, all_mu, all_sigma, k):
#     mu_k = all_mu[k]
#     sigma_k = all_sigma[k]
#     pi_k = all_pi[k]
#
#     y = multivariate_normal.pdf(X, mean=mu_k, cov=sigma_k)
#     p_Xk = y * pi_k
#     return p_Xk.reshape(-1, 1)
#
#
# # tested
# def _calculate_prob_X(X, all_pi, all_mu, all_sigma):
#     """
#     Calculates P(X), the denominator in the inference equation. Uses the following formula:
#
#     P(X^n) = SUM_k=1^K p(X^n|c^n=k) * p(c^n=k)
#
#     :param X:
#     :param all_pi:
#     :param all_mu:
#     :param all_sigma:
#     :return:
#     """
#     N = len(X)
#     K = len(all_pi)
#     all_prob = np.zeros((N, K))
#
#     for k in range(K):
#         prob_Xk = _calculate_prob_Xk(X, all_pi, all_mu, all_sigma, k)
#         all_prob[:, k] = prob_Xk[:, 0]
#     p_X = np.sum(all_prob, axis=1).reshape(-1, 1)
#     return p_X
#
#
# # tested
# def calculate_log_likelihood(X, all_pi, all_mu, all_sigma):
#     p_X = _calculate_prob_X(X, all_pi, all_mu, all_sigma)
#     log_p_X = np.log(p_X)
#     sum_p_X = np.sum(log_p_X,axis=0)
#     return sum_p_X.item(0)
#
#
# # tested
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
#         A[:, k] = a_k[:, 0]
#
#     return A



