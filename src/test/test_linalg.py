import numpy as np
import helmholtz as hm
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
