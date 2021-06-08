import helmholtz as hm
import helmholtz.setup
import helmholtz.solve.run

import numpy as np
import pytest
import scipy.sparse
import unittest


class TestMockCycle(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        hm.logging.set_simple_logging()

    def test_mock_cycle_keeps_coarse_vars_invariant_representative(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = helmholtz.setup.multilevel.Level.create_finest_level(a)
        relaxer = lambda x, b: level.relax(x, b)
        r = _create_svd_coarsening(level)

        mock_cycle = helmholtz.solve.mock_cycle.MockCycle(relaxer, r, 2)

        x = helmholtz.solve.run.random_test_matrix((n,), num_examples=3)
        x_new = mock_cycle(x)

        assert hm.linalg.scaled_norm(r.dot(x_new)) <= 1e-15

    def test_mock_cycle_svd_coarsening_faster_than_pointwise_coarsening(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = helmholtz.setup.multilevel.Level.create_finest_level(a)
        relaxer = lambda x, b: level.relax(x, b)
        r = _create_svd_coarsening(level)
        r_pointwise = _create_pointwise_coarsening(level)

        def mock_cycle_conv_factor(r, num_relax_sweeps):
            mock_cycle = helmholtz.solve.mock_cycle.MockCycle(relaxer, r, num_relax_sweeps)
            x = helmholtz.solve.run.random_test_matrix((n,), num_examples=1)
            x, conv_factor = helmholtz.solve.run.run_iterative_method(level.operator, mock_cycle, x,  num_sweeps=10)
            return conv_factor

        assert mock_cycle_conv_factor(r, 1) == pytest.approx(0.28, 1e-2)
        assert mock_cycle_conv_factor(r, 2) == pytest.approx(0.194, 1e-2)
        assert mock_cycle_conv_factor(r, 3) == pytest.approx(0.0725, 1e-2)

        assert mock_cycle_conv_factor(r_pointwise, 1) == pytest.approx(0.53, 1e-2)
        assert mock_cycle_conv_factor(r_pointwise, 2) == pytest.approx(0.56, 1e-2)
        assert mock_cycle_conv_factor(r_pointwise, 3) == pytest.approx(0.49, 1e-2)

    def test_mock_cycle_svd_coarsening_conv_factor_improves_with_num_sweeps(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = helmholtz.setup.multilevel.Level.create_finest_level(a)
        relaxer = lambda x, b: level.relax(x, b)
        r = _create_svd_coarsening(level)

        def mock_cycle_conv_factor(num_relax_sweeps):
            mock_cycle = helmholtz.solve.mock_cycle.MockCycle(relaxer, r, num_relax_sweeps)
            x = helmholtz.solve.run.random_test_matrix((n,), num_examples=1)
            x, conv_factor = helmholtz.solve.run.run_iterative_method(level.operator, mock_cycle, x, num_sweeps=10)
            return conv_factor

        assert mock_cycle_conv_factor(1) == pytest.approx(0.28, 1e-2)
        assert mock_cycle_conv_factor(2) == pytest.approx(0.194, 1e-2)
        assert mock_cycle_conv_factor(3) == pytest.approx(0.073, 1e-2)

    def test_mock_cycle_karsten_level_2_opereator(self):
        """Tests Karsten Kahl's obtained 2-level operator, which showed slow 2-level cycle that nonetheless
        improved with #sweeps."""
        a = np.array([-0.003, -0.008, -.138, -.209, 0.096, 0,     -0.138, .209, -0.003, 0.008])
        b = np.array([ 0.008,  0.020, 0.209, 0.271, 0,     0.861, -0.209, .271, -0.008, 0.020])
        n = 80
        A = np.zeros((n, n))
        for i in range(0, n, 2):
            for j in range(i - 4, i + 5):
                A[i, j % n] = a[j - i + 4]
        for i in range(1, n, 2):
            for j in range(i - 5, i + 4):
                A[i, j % n] = b[j - i + 5]
        A = scipy.sparse.csr_matrix(A)

        level = helmholtz.setup.multilevel.Level.create_finest_level(A)
        relaxer = lambda x, b: level.relax(x, b)
        r = _create_svd_coarsening(level)

        def mock_cycle_conv_factor(r, num_relax_sweeps):
            mock_cycle = helmholtz.solve.mock_cycle.MockCycle(relaxer, r, num_relax_sweeps)
            x = helmholtz.solve.run.random_test_matrix((n,), num_examples=1)
            x, conv_factor = helmholtz.solve.run.run_iterative_method(level.operator, mock_cycle, x,  num_sweeps=10)
            return conv_factor

        assert mock_cycle_conv_factor(r, 1) == pytest.approx(0.192, 1e-2)
        assert mock_cycle_conv_factor(r, 2) == pytest.approx(0.0746, 1e-2)
        assert mock_cycle_conv_factor(r, 3) == pytest.approx(0.0644, 1e-2)


def _create_svd_coarsening(level, threshold: float = 0.1):
    # Generate relaxed test matrix.
    n = level.a.shape[0]
    x = helmholtz.solve.run.random_test_matrix((n,))
    lam = 0
    b = np.zeros_like(x)
    x, conv_factor = helmholtz.solve.run.run_iterative_method(level.operator, lambda x: level.relax(x, b), x, num_sweeps=10)
    # Generate coarse variables (R) based on a window of x.
    aggregate_size = 4
    x_aggregate_t = x[:aggregate_size].transpose()
    r, _ = hm.setup.coarsening.create_coarsening(x_aggregate_t, threshold)

    # Convert to sparse matrix + tile over domain.
    r_csr = r.tile(n // aggregate_size)
    return r_csr


def _create_pointwise_coarsening(level):
    aggregate_size = 2
    r = hm.setup.coarsening.Coarsener(np.array([[1, 0]]))
    # Convert to sparse matrix + tile over domain.
    domain_size = level.a.shape[0]
    r_csr = r.tile(domain_size // aggregate_size)
    return r_csr
