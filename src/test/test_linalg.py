import numpy as np
import pytest
import scipy.linalg
import scipy.sparse
from numpy.linalg import norm
from numpy.ma.testutils import assert_array_equal, assert_array_almost_equal

import helmholtz as hm


class TestLinalg:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.random.seed(0)

    def test_scaled_norm_of_matrix_2d(self):
        num_functions = 4
        x = 2 * np.random.random((10, num_functions)) - 1

        actual = hm.linalg.scaled_norm_of_matrix(x)

        y = x.reshape(10, num_functions)
        factor = 1 / x.shape[0] ** 0.5
        expected = factor * np.array([norm(y[:, i]) for i in range(y.shape[1])])
        assert_array_almost_equal(actual, expected, 10)

    def test_scaled_norm_of_matrix_3d(self):
        num_functions = 4
        x = 2 * np.random.random((10, 2, num_functions)) - 1

        actual = hm.linalg.scaled_norm_of_matrix(x)

        y = x.reshape(20, num_functions)
        factor = 1 / np.prod(x.shape[:-1]) ** 0.5
        expected = factor * np.array([norm(y[:, i]) for i in range(y.shape[1])])
        assert_array_almost_equal(actual, expected, 10)

    def test_sparse_circulant(self):
        vals = np.array([1, 2, 3, 5, 4])
        offsets = np.array([-2, -1, 0, 2, 1])
        n = 8

        a = hm.linalg.sparse_circulant(vals, offsets, n)
        a_expected = \
            [[3, 4, 5, 0, 0, 0, 1, 2],
             [2, 3, 4, 5, 0, 0, 0, 1],
             [1, 2, 3, 4, 5, 0, 0, 0],
             [0, 1, 2, 3, 4, 5, 0, 0],
             [0, 0, 1, 2, 3, 4, 5, 0],
             [0, 0, 0, 1, 2, 3, 4, 5],
             [5, 0, 0, 0, 1, 2, 3, 4],
             [4, 5, 0, 0, 0, 1, 2, 3]]

        assert_array_almost_equal(a.toarray(), a_expected)

    def test_helmholtz_1d_operator(self):
        kh = 1.5
        n = 8
        a = hm.linalg.helmholtz_1d_operator(kh, n)

        a_expected = \
            [[0.25, 1, 0, 0, 0, 0, 0, 1],
             [1, 0.25, 1, 0, 0, 0, 0, 0],
             [0, 1, 0.25, 1, 0, 0, 0, 0],
             [0, 0, 1, 0.25, 1, 0, 0, 0],
             [0, 0, 0, 1, 0.25, 1, 0, 0],
             [0, 0, 0, 0, 1, 0.25, 1, 0],
             [0, 0, 0, 0, 0, 1, 0.25, 1],
             [1, 0, 0, 0, 0, 0, 1, 0.25]]
        assert_array_almost_equal(a.toarray(), a_expected)

    def test_stencil_grid_laplace_1d_periodic(self):
        d = 1
        n = 6
        a = hm.linalg.stencil_grid([2 * d] + [-1] * (2 * d),
                                   [np.zeros((d,), dtype=int)] +
                                   [hm.linalg.unit_vector(d, i, dtype=int) for i in range(d)] +
                                   [-hm.linalg.unit_vector(d, i, dtype=int) for i in range(d)],
                                   tuple([n] * d))
        a_expected = \
            [[2, -1, 0, 0, 0, -1],
             [-1, 2, -1, 0, 0, 0],
             [0, -1, 2, -1, 0, 0],
             [0, 0, -1, 2, -1, 0],
             [0, 0, 0, -1, 2, -1],
             [-1, 0, 0, 0, -1, 2]]
        assert_array_almost_equal(a.toarray(), a_expected)

    def test_stencil_grid_laplace_1d_dirichlet(self):
        d = 1
        n = 6
        a = hm.linalg.stencil_grid([2 * d] + [-1] * (2 * d),
                                   [np.zeros((d,), dtype=int)] +
                                   [hm.linalg.unit_vector(d, i, dtype=int) for i in range(d)] +
                                   [-hm.linalg.unit_vector(d, i, dtype=int) for i in range(d)],
                                   tuple([n] * d),
                                   boundary="dirichlet")
        a_expected = \
            [[2, -1, 0, 0, 0, 0],
             [-1, 2, -1, 0, 0, 0],
             [0, -1, 2, -1, 0, 0],
             [0, 0, -1, 2, -1, 0],
             [0, 0, 0, -1, 2, -1],
             [0, 0, 0, 0, -1, 2]]
        assert_array_almost_equal(a.toarray(), a_expected)

    def test_stencil_grid_laplace_2d_periodic(self):
        d = 2
        n = 4
        a = hm.linalg.stencil_grid([2 * d] + [-1] * (2 * d),
                                   [np.zeros((d,), dtype=int)] +
                                   [hm.linalg.unit_vector(d, i, dtype=int) for i in range(d)] +
                                   [-hm.linalg.unit_vector(d, i, dtype=int) for i in range(d)],
                                   tuple([n] * d))
        a_expected = \
            [[4, -1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
              [-1, 4, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
              [0, -1, 4, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0],
              [-1, 0, -1, 4, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1],
              [-1, 0, 0, 0, 4, -1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0],
              [0, -1, 0, 0, -1, 4, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
              [0, 0, -1, 0, 0, -1, 4, -1, 0, 0, -1, 0, 0, 0, 0, 0],
              [0, 0, 0, -1, -1, 0, -1, 4, 0, 0, 0, -1, 0, 0, 0, 0],
              [0, 0, 0, 0, -1, 0, 0, 0, 4, -1, 0, -1, -1, 0, 0, 0],
              [0, 0, 0, 0, 0, -1, 0, 0, -1, 4, -1, 0, 0, -1, 0, 0],
              [0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 4, -1, 0, 0, -1, 0],
              [0, 0, 0, 0, 0, 0, 0, -1, -1, 0, -1, 4, 0, 0, 0, -1],
              [-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 4, -1, 0, -1],
              [0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 4, -1, 0],
              [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 4, -1],
              [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, -1, 4]]
        assert_array_almost_equal(a.toarray(), a_expected)

    def test_stencil_grid_laplace_2d_dirichlet(self):
        d = 2
        n = 4
        a = hm.linalg.stencil_grid([2 * d] + [-1] * (2 * d),
                                   [np.zeros((d,), dtype=int)] +
                                   [hm.linalg.unit_vector(d, i, dtype=int) for i in range(d)] +
                                   [-hm.linalg.unit_vector(d, i, dtype=int) for i in range(d)],
                                   tuple([n] * d),
                                   boundary="dirichlet")
        a_expected = \
            [[4, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-1, 4, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, -1, 4, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, -1, 4, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
             [-1, 0, 0, 0, 4, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
             [0, -1, 0, 0, -1, 4, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
             [0, 0, -1, 0, 0, -1, 4, -1, 0, 0, -1, 0, 0, 0, 0, 0],
             [0, 0, 0, -1, 0, 0, -1, 4, 0, 0, 0, -1, 0, 0, 0, 0],
             [0, 0, 0, 0, -1, 0, 0, 0, 4, -1, 0, 0, -1, 0, 0, 0],
             [0, 0, 0, 0, 0, -1, 0, 0, -1, 4, -1, 0, 0, -1, 0, 0],
             [0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 4, -1, 0, 0, -1, 0],
             [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 4, 0, 0, 0, -1],
             [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 4, -1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 4, -1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 4, -1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 4]]
        assert_array_almost_equal(a.toarray(), a_expected)

    def test_falgout_uxxyy(self):
        stencil = [1, -2, 1,
                   -2, 4, -2,
                   1, -2, 1]
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1), (0, 0), (0, 1),
                   (1, -1), (1, 0), (1, 1),]
        a =  hm.linalg.stencil_grid(stencil, offsets, (4, 4))
        a_expected = \
            [[4, -2, 0, -2, -2, 1, 0, 1, 0, 0, 0, 0, -2, 1, 0, 1],
             [-2, 4, -2, 0, 1, -2, 1, 0, 0, 0, 0, 0, 1, -2, 1, 0],
             [0, -2, 4, -2, 0, 1, -2, 1, 0, 0, 0, 0, 0, 1, -2, 1],
             [-2, 0, -2, 4, 1, 0, 1, -2, 0, 0, 0, 0, 1, 0, 1, -2],
             [-2, 1, 0, 1, 4, -2, 0, -2, -2, 1, 0, 1, 0, 0, 0, 0],
             [1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1, 0, 0, 0, 0, 0],
             [0, 1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1, 0, 0, 0, 0],
             [1, 0, 1, -2, -2, 0, -2, 4, 1, 0, 1, -2, 0, 0, 0, 0],
             [0, 0, 0, 0, -2, 1, 0, 1, 4, -2, 0, -2, -2, 1, 0, 1],
             [0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1, 0],
             [0, 0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2, 0, 1, -2, 1],
             [0, 0, 0, 0, 1, 0, 1, -2, -2, 0, -2, 4, 1, 0, 1, -2],
             [-2, 1, 0, 1, 0, 0, 0, 0, -2, 1, 0, 1, 4, -2, 0, -2],
             [1, -2, 1, 0, 0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2, 0],
             [0, 1, -2, 1, 0, 0, 0, 0, 0, 1, -2, 1, 0, -2, 4, -2],
             [1, 0, 1, -2, 0, 0, 0, 0, 1, 0, 1, -2, -2, 0, -2, 4]]
        assert_array_almost_equal(a.toarray(), a_expected)

    def test_falgout_operator(self):
        n = 4
        h = 1 / n
        a = hm.linalg.falgout_mixed_elliptic(1, 10, 1, (n, n), h)
        a_expected = \
            [[1376, -672, 0, -672, -528, 256, 0, 256, 0, 0, 0, 0, -528, 256, 0, 256],
             [-672, 1376, -672, 0, 256, -528, 256, 0, 0, 0, 0, 0, 256, -528, 256, 0],
             [0, -672, 1376, -672, 0, 256, -528, 256, 0, 0, 0, 0, 0, 256, -528, 256],
             [-672, 0, -672, 1376, 256, 0, 256, -528, 0, 0, 0, 0, 256, 0, 256, -528],
             [-528, 256, 0, 256, 1376, -672, 0, -672, -528, 256, 0, 256, 0, 0, 0, 0],
             [256, -528, 256, 0, -672, 1376, -672, 0, 256, -528, 256, 0, 0, 0, 0, 0],
             [0, 256, -528, 256, 0, -672, 1376, -672, 0, 256, -528, 256, 0, 0, 0, 0],
             [256, 0, 256, -528, -672, 0, -672, 1376, 256, 0, 256, -528, 0, 0, 0, 0],
             [0, 0, 0, 0, -528, 256, 0, 256, 1376, -672, 0, -672, -528, 256, 0, 256],
             [0, 0, 0, 0, 256, -528, 256, 0, -672, 1376, -672, 0, 256, -528, 256, 0],
             [0, 0, 0, 0, 0, 256, -528, 256, 0, -672, 1376, -672, 0, 256, -528, 256],
             [0, 0, 0, 0, 256, 0, 256, -528, -672, 0, -672, 1376, 256, 0, 256, -528],
             [-528, 256, 0, 256, 0, 0, 0, 0, -528, 256, 0, 256, 1376, -672, 0, -672],
             [256, -528, 256, 0, 0, 0, 0, 0, 256, -528, 256, 0, -672, 1376, -672, 0],
             [0, 256, -528, 256, 0, 0, 0, 0, 0, 256, -528, 256, 0, -672, 1376, -672],
             [256, 0, 256, -528, 0, 0, 0, 0, 256, 0, 256, -528, -672, 0, -672, 1376]]
        assert_array_almost_equal(a.toarray(), a_expected)

    def test_tile_csr_matrix(self):
        n = 4
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)

        a_expected = \
            [[-1.75, 1, 0, 1],
             [1, -1.75, 1, 0],
             [0, 1, -1.75, 1],
             [1, 0, 1, -1.75]]
        assert_array_almost_equal(a.toarray(), a_expected)

        for growth_factor in range(2, 5):
            a_tiled = hm.linalg.tile_csr_matrix(a, growth_factor)
            a_tiled_expected = hm.linalg.helmholtz_1d_operator(kh, growth_factor * n).tocsr()
            assert_array_almost_equal(a_tiled.toarray(), a_tiled_expected.toarray())

    def test_tile_csr_matrix_level2_operator(self):
        a = np.array([
            [-0.16, 0.05, 0.18, 0.35, 0, 0, 0.18, -0.21],
            [0.05, -1.22, -0.21, -0.42, 0, 0, 0.35, -0.42],
            [0.18, -0.21, -0.16, 0.05, 0.18, 0.35, 0, 0],
            [0.35, -0.42, 0.05, -1.22, -0.21, -0.42, 0, 0],
            [0, 0, 0.18, -0.21, -0.16, 0.05, 0.18, 0.35],
            [0, 0, 0.35, -0.42, 0.05, -1.22, -0.21, -0.42],
            [0.18, 0.35, 0, 0, 0.18, -0.21, -0.16, 0.05],
            [-0.21, -0.42, 0, 0, 0.35, -0.42, 0.05, -1.22]
        ])

        a_tiled = hm.linalg.tile_csr_matrix(scipy.sparse.csr_matrix(a), 2)

        # The tiled operator should have blocks of of 2 6-point stencils that are periodic on 16 points.
        wrapped_around = np.zeros_like(a)
        wrapped_around[-2:, :2] = a[-2:, :2]
        wrapped_around[:2, -2:] = a[:2, -2:]
        a_tiled_expected = np.block([[a - wrapped_around, wrapped_around], [wrapped_around, a - wrapped_around]])
        assert_array_almost_equal(a_tiled.toarray(), a_tiled_expected)

    def test_tile_array(self):
        a = np.array([[1, 2], [3, 4], [5, 6]])

        a_tiled = hm.linalg.tile_array(a, 2)

        a_tiled_expected = np.array([[1, 2, 0, 0],
                                     [3, 4, 0, 0],
                                     [5, 6, 0, 0],
                                     [0, 0, 1, 2],
                                     [0, 0, 3, 4],
                                     [0, 0, 5, 6]])
        assert_array_almost_equal(a_tiled.toarray(), a_tiled_expected)

    def test_gram_schmidt(self):
        a = np.random.random((10, 4))

        q = hm.linalg.gram_schmidt(a)

        q_expected = _gram_schmidt_explicit(a)
        assert_array_almost_equal(hm.linalg.normalize_signs(q, axis=1),
                                  hm.linalg.normalize_signs(q_expected, axis=1))

    def test_ritz(self):
        n = 16
        kh = 0
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x = np.random.random((n, 4))
        action = lambda x: a.dot(x)

        y, lam_y = hm.linalg.ritz(x, action)

        lam_y_expected = np.array([_rq(y[:, k], action) for k in range(y.shape[1])])
        assert_array_almost_equal(lam_y, lam_y_expected)

        assert y.shape == x.shape
        r = action(y)
        r = np.array([r[:, k] - lam_y[k] * y[:, k] for k in range(y.shape[1])])

        assert norm(r.dot(x)) < 1e-14

    def test_sparse_lu_solver(self):
        # Construct an aggregation matrix Q (has at least one non-zero per row).
        m, n = 20, 50
        p = scipy.sparse.csr_matrix((np.ones(m), (np.arange(m), np.random.choice(np.arange(n), m))), shape=(m, n))
        q = p + scipy.sparse.random(m, n, density=0.05)
        # A = Q*Q^T.
        a = q.dot(q.T)

        lu_solver = hm.linalg.SparseLuSolver(a)
        b = np.random.random((m, 5))

        x = lu_solver.solve(b)

        assert norm(a.dot(x) - b) <= 1e-7 * norm(b)

    def test_pairwise_cos_similarity(self):
        # Compare with explicit summation.
        x = 2 * np.random.random((10, 4)) - 1
        y = 2 * np.random.random((10, 3)) - 1
        x[:, 0] = 0

        actual = hm.linalg.pairwise_cos_similarity(x, y)
        expected = np.array([
            [sum(x[:, i] * y[:, j]) / np.clip((sum(x[:, i] ** 2) * sum(y[:, j] ** 2)) ** 0.5, 1e-15, None)
             for j in range(y.shape[1])]
            for i in range(x.shape[1])])

        assert_array_almost_equal(actual, expected, 10)

    def test_create_folds(self):
        x = 2 * np.random.random((10, 4)) - 1

        folds = hm.linalg.create_folds(x, (2, 3, 5))

        assert_array_equal(folds[0], x[0:2])
        assert_array_equal(folds[1], x[2:5])
        assert_array_equal(folds[2], x[5:10])

    def test_get_windows_by_index(self):
        n = 10
        num_functions = 4
        aggregate_size = 4
        stride = 2
        num_windows = 21
        x = np.arange(num_functions * n).reshape(num_functions, n).T

        x_windows = hm.linalg.get_windows_by_index(x, np.arange(aggregate_size), stride, num_windows)

        assert x_windows.shape == (num_windows, aggregate_size)
        assert np.array_equal(x_windows, np.array([
             [0, 1, 2, 3],
             [10, 11, 12, 13],
             [20, 21, 22, 23],
             [30, 31, 32, 33],
             [2, 3, 4, 5],
             [12, 13, 14, 15],
             [22, 23, 24, 25],
             [32, 33, 34, 35],
             [4, 5, 6, 7],
             [14, 15, 16, 17],
             [24, 25, 26, 27],
             [34, 35, 36, 37],
             [6, 7, 8, 9],
             [16, 17, 18, 19],
             [26, 27, 28, 29],
             [36, 37, 38, 39],
             [8, 9, 0, 1],
             [18, 19, 10, 11],
             [28, 29, 20, 21],
             [38, 39, 30, 31],
             [0, 1, 2, 3]
        ]))


def _rq(x, action):
    """
    Returns the Rayleigh-Quotient (x,action(x))/(x,x) of the vector x.
    """
    return (action(x)).dot(x) / (x.dot(x))


def _gram_schmidt_explicit(a: np.ndarray) -> np.ndarray:
    """
    Performs a Gram-Schmidt orthonormalization on matrix columns. Does not use the QR factorization but an
    explicit implementation.

    Args:
        a: original matrix.

    Returns: orthonormalized matrix a.
    """
    a = a.copy()
    a[:, 0] /= norm(a[:, 0])
    for i in range(1, a.shape[1]):
        ai = a[:, i]
        ai -= sum((ai.dot(a[:, j])) * a[:, j] for j in range(i))
        a[:, i] = ai / norm(ai)
    return a
