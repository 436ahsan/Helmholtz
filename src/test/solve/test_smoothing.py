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
        factor, num_sweeps, _, _, _, conv = \
            hm.solve.smoothing.shrinkage_factor(operator, lambda x, b: kaczmarz.step(x, b), (n, ))
        assert factor == pytest.approx(0.62, 1e-2)
        assert num_sweeps == 5
        assert conv == pytest.approx(0.91, 1e-2)

        # GS is a more efficient smoother, thus takes less to slow down.
        gs = hm.solve.relax.GsRelaxer(a)
        factor, num_sweeps, _, _, _, conv = \
            hm.solve.smoothing.shrinkage_factor(operator, lambda x, b: gs.step(x, b), (n, ))
        assert factor == pytest.approx(0.43, 1e-2)
        assert num_sweeps == 4
        assert conv == pytest.approx(0.85, 1e-2)

    def test_shrinkage_factor_helmholtz(self):
        n = 96
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        operator = lambda x: a.dot(x)

        kaczmarz = hm.solve.relax.KaczmarzRelaxer(a, scipy.sparse.eye(a.shape[0]))
        factor, num_sweeps, _, _, _, conv = \
            hm.solve.smoothing.shrinkage_factor(operator, lambda x, b: kaczmarz.step(x, b), (n, ))
        assert factor == pytest.approx(0.63, 1e-2)
        assert num_sweeps == 5
        assert conv == pytest.approx(0.93, 1e-2)

        # GS is more efficient than Kaczmarz here too, but diverges.
        gs = hm.solve.relax.GsRelaxer(a)
        factor, num_sweeps, _, _, _, conv = \
            hm.solve.smoothing.shrinkage_factor(operator, lambda x, b: gs.step(x, b), (n, ))
        assert factor == pytest.approx(0.48, 1e-2)
        assert num_sweeps == 3
        assert conv == pytest.approx(1.3, 1e-2)
