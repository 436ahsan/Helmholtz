import helmholtz as hm
import itertools
import logging
import numpy as np
import pytest
import scipy.sparse
import unittest
from numpy.linalg import norm
from numpy.ma.testutils import assert_array_almost_equal


class TestMockCycle(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        hm.logging.set_simple_logging()

    def test_mock_cycle_keeps_coarse_vars_invariant_representative(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = hm.multilevel.Level.create_finest_level(a)
        relaxer = lambda x, b: level.relax(x, b, lam=0)
        r = _create_svd_coarsening(level)

        mock_cycle = hm.mock_cycle.MockCycle(relaxer, r, 2)

        x = hm.run.random_test_matrix((n,), num_examples=3)
        x_new = mock_cycle(x)
        #x, _, _ = hm.run.run_iterative_method(level.operator, mock_cycle, x, lam, num_sweeps=num_sweeps)

        assert hm.linalg.scaled_norm(r.dot(x_new)) <= 1e-15

    def test_mock_cycle_svd_coarsening_faster_than_pointwise_coarsening(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = hm.multilevel.Level.create_finest_level(a)
        relaxer = lambda x, b: level.relax(x, b, lam=0)
        r = _create_svd_coarsening(level)
        r_pointwise = _create_pointwise_coarsening(level.a.shape[0])

        def mock_cycle_conv_factor(r, num_relax_sweeps):
            mock_cycle = hm.mock_cycle.MockCycle(relaxer, r, num_relax_sweeps)
            x = hm.run.random_test_matrix((n,), num_examples=1)
            lam = 0
            x, _, conv_factor = hm.run.run_iterative_method(
                level.operator, lambda x, lam: (mock_cycle(x), lam), x, lam, num_sweeps=10)
            return conv_factor

        assert mock_cycle_conv_factor(r, 1) == pytest.approx(0.28, 1e-2)
        assert mock_cycle_conv_factor(r, 2) == pytest.approx(0.194, 1e-2)
        assert mock_cycle_conv_factor(r, 3) == pytest.approx(0.074, 1e-2)

        assert mock_cycle_conv_factor(r_pointwise, 1) == pytest.approx(0.53, 1e-2)
        assert mock_cycle_conv_factor(r_pointwise, 2) == pytest.approx(0.56, 1e-2)
        assert mock_cycle_conv_factor(r_pointwise, 3) == pytest.approx(0.49, 1e-2)

    def test_mock_cycle_svd_coarsening_conv_factor_improves_with_num_sweeps(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = hm.multilevel.Level.create_finest_level(a)
        relaxer = lambda x, b: level.relax(x, b, lam=0)
        r = _create_svd_coarsening(level)

        def mock_cycle_conv_factor(num_relax_sweeps):
            mock_cycle = hm.mock_cycle.MockCycle(relaxer, r, num_relax_sweeps)
            x = hm.run.random_test_matrix((n,), num_examples=1)
            lam = 0
            x, _, conv_factor = hm.run.run_iterative_method(
                level.operator, lambda x, lam: (mock_cycle(x), lam), x, lam, num_sweeps=10)
            return conv_factor

        assert mock_cycle_conv_factor(1) == pytest.approx(0.28, 1e-2)
        assert mock_cycle_conv_factor(2) == pytest.approx(0.194, 1e-2)
        assert mock_cycle_conv_factor(3) == pytest.approx(0.074, 1e-2)

def _create_svd_coarsening(level):
    # Generate relaxed test matrix.
    n = level.a.shape[0]
    x = hm.run.random_test_matrix((n,))
    lam = 0
    b = np.zeros_like(x)
    x, _, _ = hm.run.run_iterative_method(level.operator, lambda x, lam: (level.relax(x, b, lam), lam),
                                          x, lam, num_sweeps=10)
    # Generate coarse variables (R) based on a window of x.
    aggregate_size = 4
    x_aggregate_t = x[:aggregate_size].transpose()
    r, _ = hm.coarsening.create_coarsening(x_aggregate_t, 0.1)

    # Convert to sparse matrix + tile over domain.
    r_csr = r.tile(n // aggregate_size)
    return r_csr


def _create_pointwise_coarsening(domain_size):
    aggregate_size = 2
    r = hm.coarsening.Coarsener(np.array([[1, 0]]))
    # Convert to sparse matrix + tile over domain.
    r_csr = r.tile(domain_size // aggregate_size)
    return r_csr
