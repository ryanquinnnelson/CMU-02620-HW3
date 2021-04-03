import numpy as np
from packages.mixture import GaussianMixture as gm


def test__initialize_assignments():
    X = np.array([[6, 4],
                  [4, 1],
                  [1, 2],
                  [2, 1]])

    K = 3

    actual = gm._initialize_assignments(X, K)

    # ensure each row adds up to 1.0
    expected_column_totals = np.array([[1.0],
                                       [1.0],
                                       [1.0],
                                       [1.0]])
    np.testing.assert_allclose(np.sum(actual, axis=1).reshape(-1, 1), expected_column_totals, atol=1e-16)
    assert actual.shape == (4, 3)


def test___init__():
    K = 3
    epsilon = 1e-3

    model = gm.GaussianMixture(K, epsilon)

    assert model.K == K
    assert model.epsilon == epsilon
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
