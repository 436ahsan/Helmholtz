import numpy as np
import helmholtz as hm
import scipy.sparse
from numpy.ma.testutils import assert_array_almost_equal


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

    def test_periodic_tile_square_matrix(self):
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
            a_tiled = hm.linalg.periodic_tile(a, growth_factor)
            a_tiled_expected = hm.linalg.helmholtz_1d_operator(kh, growth_factor * n).tocsr()
            assert_array_almost_equal(a_tiled.toarray(), a_tiled_expected.toarray())

    def test_periodic_tile_dense_matrix(self):
        a = np.array([[1, 2], [3, 4], [5, 6]])

        a_tiled = hm.linalg.tile_dense(a, 2)

        print(np.array2string(a_tiled.toarray(), separator=","))

        a_tiled_expected = np.array([[1, 2, 0, 0],
             [3, 4, 0, 0],
             [5, 6, 0, 0],
             [0, 0, 1, 2],
             [0, 0, 3, 4],
             [0, 0, 5, 6]])
        assert_array_almost_equal(a_tiled.toarray(), a_tiled_expected)