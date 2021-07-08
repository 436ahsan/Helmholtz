import numpy as np
import logging
import pytest
import scipy.sparse
import unittest
from numpy.ma.testutils import assert_array_almost_equal

import helmholtz as hm


class TestSmoothing(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        hm.logging.set_simple_logging(logging.INFO)

    def test_shrinkage_factor_laplace(self):
        n = 96
        kh = 0
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        operator = lambda x: a.dot(x)

        kaczmarz = hm.solve.relax.KaczmarzRelaxer(a, scipy.sparse.eye(a.shape[0]))
        p, conv = hm.solve.smoothing.shrinkage_factor(operator, kaczmarz, (n, ))
        assert_array_almost_equal(p, [3., 0.69307541,  1.86489, -0.16283799])

        # GS is a more efficient smoother and takes longer to slow down.
        gs = hm.solve.relax.GsRelaxer(a)
        p, conv = hm.solve.smoothing.shrinkage_factor(operator, gs, (n, ), print_frequency=1, max_sweeps=5)
        assert_array_almost_equal(p, [4.03305045,  0.56684545,  1.68422643, -0.33782641])

    def test_shrinkage_factor_helmholtz(self):
        n = 96
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        operator = lambda x: a.dot(x)

        kaczmarz = hm.solve.relax.KaczmarzRelaxer(a, scipy.sparse.eye(a.shape[0]))
        p, conv = hm.solve.smoothing.shrinkage_factor(operator, kaczmarz, (n, ), slow_conv_factor=0.99)
        assert_array_almost_equal(p, [ 3.666572,  0.691874,  1.003036, -1.535012])

        # GS is more efficient than Kaczmarz here too, but diverges.
        gs = hm.solve.relax.GsRelaxer(a)
        p, conv = hm.solve.smoothing.shrinkage_factor(operator, gs, (n, ), slow_conv_factor=1.1)
        assert_array_almost_equal(p, [ 3.639491,  0.520149,  1.000983, -8.711164])
