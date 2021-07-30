import logging
import pytest
import sys
import numpy as np
from numpy.ma.testutils import assert_array_almost_equal

import helmholtz as hm

logger = logging.getLogger("nb")


class TestCoarseningUniform:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.set_printoptions(precision=6, linewidth=1000)
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(stream=sys.stdout, level=logging.WARN, format="%(levelname)-8s %(message)s",
                            datefmt="%a, %d %b %Y %H:%M:%S")
        np.random.seed(1)

    def test_create_uniform_coarsening_domain(self):
        n = 16
        kh = 0.5
        aggregate_size = 4
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10)

        # Generate coarse variables (R) on the non-repetitive domain.
        result = list(hm.setup.coarsening_uniform.create_coarsening_domain_uniform(x, aggregate_size))

        # For a cycle index of 1, the max coarsening ratio is 0.7, and aggregate_size * 0.7 = 2.8, so we can have at
        # most # 2 PCs per aggregate.
        assert len(result) == 2
        r_values = np.array([item[0] for item in result])
        mean_energy_error = np.array([item[1] for item in result])

        assert_array_almost_equal(mean_energy_error, [0.48278, 0.139669], decimal=5)
        assert r_values[0].shape == (4, 16)
        r = r_values[1]
        assert r.shape == (8, 16)
        # To print r:
        # print(','.join(np.array2string(y, separator=",", formatter={'float_kind':lambda x: "%.2f" % x})
        # for y in np.array(r.todense())))
        assert_array_almost_equal(
            r.todense(), [
                [0.47,0.61,0.54,0.34,0,0,0,0,0,0,0,0,0,0,0,0],
                [-0.64,-0.18,0.34,0.67,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0.34,0.54,0.60,0.49,0,0,0,0,0,0,0,0],
                [0,0,0,0,0.70,0.32,-0.19,-0.61,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0.47,0.58,0.54,0.38,0,0,0,0],
                [0,0,0,0,0,0,0,0,-0.65,-0.20,0.31,0.66,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,-0.39,-0.53,-0.57,-0.50],
                [0,0,0,0,0,0,0,0,0,0,0,0,-0.67,-0.33,0.20,0.64]
            ], decimal=2)

    def test_create_uniform_coarsening_domain_indivisible_size(self):
        n = 17
        kh = 0.5
        aggregate_size = 4
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10)

        # Generate coarse variables (R) on the non-repetitive domain.
        result = list(hm.setup.coarsening_uniform.create_coarsening_domain_uniform(x, aggregate_size))

        # For a cycle index of 1, the max coarsening ratio is 0.7, and aggregate_size * 0.7 = 2.8, so we can have at
        # most # 2 PCs per aggregate.
        assert len(result) == 2
        r_values = np.array([item[0] for item in result])
        mean_energy_error = np.array([item[1] for item in result])

        assert_array_almost_equal(mean_energy_error, [0.513766, 0.135483], decimal=5)
        assert r_values[0].shape == (5, 17)
        r = r_values[1]
        assert r.shape == (10, 17)
        # To print r:
        # print(','.join(np.array2string(y, separator=",", formatter={'float_kind':lambda x: "%.2f" % x})
        #       for y in np.array(r.todense())))
        assert_array_almost_equal(
            r.todense(), [
                [-0.47, -0.59, -0.54, -0.38, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.63, 0.21, -0.30, -0.68, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -0.40, -0.55, -0.57, -0.45, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.68, 0.27, -0.24, -0.64, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0.53, 0.61, 0.52, 0.29, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0.59, 0.14, -0.37, -0.70, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.29, 0.52, 0.61, 0.52, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.70, -0.38, 0.14, 0.59, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.37, 0.55, 0.58, 0.48],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.66, -0.33, 0.20, 0.64]
            ], decimal=2)

    def test_create_uniform_coarsening_domain_optimize_kh_0_5(self):
        n = 96
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = hm.repetitive.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10)

        # Calculate mock cycle predicted efficiency.
        aggregate_size_values = np.array([2, 4, 6])
        nu_values = np.arange(1, 6, dtype=int)
        max_conv_factor = 0.3

        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, aggregate_size_values, nu_values)
        r, aggregate_size, nc, cr, mean_energy_error, nu, mock_conv, mock_work, mock_efficiency = \
            coarsener.get_optimal_coarsening(max_conv_factor)

        assert r.shape == (48, 96)
        assert aggregate_size == 2
        assert nc == 1
        assert cr == pytest.approx(0.5, 1e-2)
        assert mean_energy_error == pytest.approx(0.256, 1e-2)
        assert nu == 2
        assert mock_conv == pytest.approx(0.222, 1e-2)
        assert mock_work == pytest.approx(4, 1e-2)
        assert mock_efficiency == pytest.approx(0.687, 1e-2)

    def test_create_uniform_coarsening_domain_optimize_kh_1(self):
        n = 96
        kh = 1
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = hm.repetitive.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10)

        # Calculate mock cycle predicted efficiency.
        aggregate_size_values = np.array([2, 4, 6])
        nu_values = np.arange(1, 6, dtype=int)
        max_conv_factor = 0.3

        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, aggregate_size_values, nu_values)
        r, aggregate_size, nc, cr, mean_energy_error, nu, mock_conv, mock_work, mock_efficiency = \
            coarsener.get_optimal_coarsening(max_conv_factor)

        assert r.shape == (48, 96)
        assert aggregate_size == 4
        assert nc == 2
        assert cr == pytest.approx(0.5, 1e-2)
        assert mean_energy_error == pytest.approx(0.156, 1e-2)
        assert nu == 3
        assert mock_conv == pytest.approx(0.156, 1e-2)
        assert mock_work == pytest.approx(6, 1e-2)
        assert mock_efficiency == pytest.approx(0.733, 1e-2)

    def test_create_uniform_coarsening_domain_optimize_kh_0_5_repetitive(self):
        n = 96
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = hm.repetitive.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10, num_examples=2)

        # Calculate mock cycle predicted efficiency.
        aggregate_size_values = np.array([2, 4, 6])
        nu_values = np.arange(1, 6, dtype=int)
        max_conv_factor = 0.3

        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, aggregate_size_values, nu_values,
                                                                 repetitive=True)
        r, aggregate_size, nc, cr, mean_energy_error, nu, mock_conv, mock_work, mock_efficiency = \
            coarsener.get_optimal_coarsening(max_conv_factor)

        assert r.shape == (48, 96)
        assert aggregate_size == 2
        assert nc == 1
        assert cr == pytest.approx(0.5, 1e-2)
        assert mean_energy_error == pytest.approx(0.191, 1e-2)
        assert nu == 2
        assert mock_conv == pytest.approx(0.222, 1e-2)
        assert mock_work == pytest.approx(4, 1e-2)
        assert mock_efficiency == pytest.approx(0.687, 1e-2)


def _get_test_matrix(a, n, num_sweeps, num_examples: int = None):
    level = hm.repetitive.hierarchy.create_finest_level(a)
    x = hm.solve.run.random_test_matrix((n,), num_examples=num_examples)
    b = np.zeros_like(x)
    x, _ = hm.solve.run.run_iterative_method(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)
    return x
