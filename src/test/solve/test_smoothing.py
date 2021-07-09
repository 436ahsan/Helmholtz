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
        factor, num_sweeps, _, _ = hm.solve.smoothing.shrinkage_factor(operator, lambda x, b: kaczmarz.step(x, b), (n, ),
                                                           print_frequency=1, max_sweeps=20)
        assert factor == pytest.approx(0.73, 1e-2)
        assert num_sweeps == 6

        # GS is a more efficient smoother, thus takes less to slow down.
        gs = hm.solve.relax.GsRelaxer(a)
        factor, num_sweeps, _, _ = hm.solve.smoothing.shrinkage_factor(operator, lambda x, b: gs.step(x, b), (n, ),
                                                            print_frequency=1, max_sweeps=20)
        assert factor == pytest.approx(0.60, 1e-2)
        assert num_sweeps == 6

    def test_shrinkage_factor_helmholtz(self):
        n = 96
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        operator = lambda x: a.dot(x)

        kaczmarz = hm.solve.relax.KaczmarzRelaxer(a, scipy.sparse.eye(a.shape[0]))
        factor, num_sweeps, _, _ = hm.solve.smoothing.shrinkage_factor(operator, lambda x, b: kaczmarz.step(x, b), (n, ),
                                                           print_frequency=1, max_sweeps=20)
        assert factor == pytest.approx(0.74, 1e-2)
        assert num_sweeps == 6

        # GS is more efficient than Kaczmarz here too, but diverges.
        gs = hm.solve.relax.GsRelaxer(a)
        factor, num_sweeps, _, _ = hm.solve.smoothing.shrinkage_factor(operator, lambda x, b: gs.step(x, b), (n, ),
                                                            print_frequency=1, max_sweeps=20)
        assert factor == pytest.approx(0.64, 1e-2)
        assert num_sweeps == 5
