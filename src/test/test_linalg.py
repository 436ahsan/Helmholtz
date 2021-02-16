import numpy as np
import scipy.linalg
import scipy.sparse
from numpy.linalg import norm
from numpy.ma.testutils import assert_array_almost_equal

import helmholtz as hm


class TestLinalg:

    def test_sparse_circulant(self):
        vals = np.array([1, 2, 3, 5, 4])
        offsets = np.array([-2, -1, 0, 2, 1])
        n = 8

        a = hm.linalg.sparse_circulant(vals, offsets, n)
        a_expected = \
            [[3., 4., 5., 0., 0., 0., 1., 2.],
             [2., 3., 4., 5., 0., 0., 0., 1.],
             [1., 2., 3., 4., 5., 0., 0., 0.],
             [0., 1., 2., 3., 4., 5., 0., 0.],
             [0., 0., 1., 2., 3., 4., 5., 0.],
             [0., 0., 0., 1., 2., 3., 4., 5.],
             [5., 0., 0., 0., 1., 2., 3., 4.],
             [4., 5., 0., 0., 0., 1., 2., 3.]]

        assert_array_almost_equal(a.toarray(), a_expected)

    def test_helmholtz_1d_operator(self):
        kh = 1.5
        n = 8
        a = hm.linalg.helmholtz_1d_operator(kh, n)

        a_expected = \
            [[0.25, 1., 0., 0., 0., 0., 0., 1.],
             [1., 0.25, 1., 0., 0., 0., 0., 0.],
             [0., 1., 0.25, 1., 0., 0., 0., 0.],
             [0., 0., 1., 0.25, 1., 0., 0., 0.],
             [0., 0., 0., 1., 0.25, 1., 0., 0.],
             [0., 0., 0., 0., 1., 0.25, 1., 0.],
             [0., 0., 0., 0., 0., 1., 0.25, 1.],
             [1., 0., 0., 0., 0., 0., 1., 0.25]]
        assert_array_almost_equal(a.toarray(), a_expected)

    def test_tile_csr_matrix(self):
        n = 4
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n).tocsr()

        a_expected = \
            [[-1.75, 1., 0., 1.],
             [1., -1.75, 1., 0.],
             [0., 1., -1.75, 1.],
             [1., 0., 1., -1.75]]
        assert_array_almost_equal(a.toarray(), a_expected)

        for growth_factor in range(2, 5):
            a_tiled = hm.linalg.tile_csr_matrix(a, growth_factor)
            a_tiled_expected = hm.linalg.helmholtz_1d_operator(kh, growth_factor * n).tocsr()
            assert_array_almost_equal(a_tiled.toarray(), a_tiled_expected.toarray())

    def test_tile_csr_matrix_level2_operator(self):
        a = np.array([
            [-0.16, 0.05, 0.18, 0.35, 0., 0., 0.18, -0.21],
            [0.05, -1.22, -0.21, -0.42, 0., 0., 0.35, -0.42],
            [0.18, -0.21, -0.16, 0.05, 0.18, 0.35, 0., 0.],
            [0.35, -0.42, 0.05, -1.22, -0.21, -0.42, 0., 0.],
            [0., 0., 0.18, -0.21, -0.16, 0.05, 0.18, 0.35],
            [0., 0., 0.35, -0.42, 0.05, -1.22, -0.21, -0.42],
            [0.18, 0.35, 0., 0., 0.18, -0.21, -0.16, 0.05],
            [-0.21, -0.42, 0., 0., 0.35, -0.42, 0.05, -1.22]
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
