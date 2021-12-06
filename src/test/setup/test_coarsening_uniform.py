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
        coarsener = hm.setup.coarsening_uniform.FixedAggSizeUniformCoarsener(x, aggregate_size)
        result = coarsener.all_candidate_coarsening()

        # For a cycle index of 1, the max coarsening ratio is 0.7, and aggregate_size * 0.7 = 2.8, so we can have at
        # most # 2 PCs per aggregate.
        assert len(result) == 2
        r_values = np.array([item[1][0] for item in result])
        mean_energy_error = np.array([item[1][1] for item in result])

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
        coarsener = hm.setup.coarsening_uniform.FixedAggSizeUniformCoarsener(x, aggregate_size, repetitive=True)
        result = coarsener.all_candidate_coarsening()

        # For a cycle index of 1, the max coarsening ratio is 0.7, and aggregate_size * 0.7 = 2.8, so we can have at
        # most # 2 PCs per aggregate.
        assert len(result) == 2
        r_values = np.array([item[1][0] for item in result])
        mean_energy_error = np.array([item[1][1] for item in result])

        assert_array_almost_equal(mean_energy_error, [0.477069, 0.175552], decimal=5)
        assert r_values[0].shape == (5, 17)
        r = r_values[1]
        assert r.shape == (10, 17)
        # To print r:
        # print(','.join(np.array2string(y, separator=",", formatter={'float_kind':lambda x: "%.2f" % x})
        #       for y in np.array(r.todense())))
        assert_array_almost_equal(
            r.todense(), [
                [-0.27, -0.53, -0.62, -0.52, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                 0.00],
                [0.70, 0.38, -0.12, -0.59, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                 0.00],
                [0.00, 0.00, 0.00, 0.00, -0.27, -0.53, -0.62, -0.52, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.70, 0.38, -0.12, -0.59, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.27, -0.53, -0.62, -0.52, 0.00, 0.00, 0.00, 0.00,
                 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.70, 0.38, -0.12, -0.59, 0.00, 0.00, 0.00, 0.00,
                 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.27, -0.53, -0.62, -0.52,
                 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.70, 0.38, -0.12, -0.59,
                 0.00],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.27, -0.53, -0.62,
                 -0.52],
                [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.70, 0.38, -0.12, -0.59]
            ], decimal=2)

    def test_create_uniform_coarsening_domain_optimize_kh_0_5(self):
        n = 96
        kh = 0.5
        max_conv_factor = 0.3

        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = hm.setup.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10)

        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, 4)
        # Calculate best mock cycle predicted efficiency.
        r, aggregate_size, num_components, cr, mean_energy_error, mock_conv, mock_work, mock_efficiency = \
            coarsener.get_optimal_coarsening(max_conv_factor)

        assert r.shape == (48, 96)
        assert aggregate_size == 4
        assert num_components == 2
        assert cr == pytest.approx(0.5, 1e-2)
        assert mean_energy_error == pytest.approx(0.128, 1e-2)
        assert mock_conv == pytest.approx(0.141, 1e-2)
        assert mock_work == pytest.approx(8, 1e-2)
        assert mock_efficiency == pytest.approx(0.782, 1e-2)

    def test_create_uniform_coarsening_domain_optimize_kh_1(self):
        """Using the current criteria, this gives a 2:3 coarsening ratio, which may or may not be too large."""
        n = 96
        kh = 1
        max_conv_factor = 0.3
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = hm.setup.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10)

        # Calculate mock cycle predicted efficiency.
        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, 4)
        r, aggregate_size, nc, cr, mean_energy_error, mock_conv, mock_work, mock_efficiency = \
            coarsener.get_optimal_coarsening(max_conv_factor)

        assert r.shape == (64, 96)
        assert aggregate_size == 3
        assert nc == 2
        assert cr == pytest.approx(0.67, 1e-2)
        assert mean_energy_error == pytest.approx(0.086, 1e-2)
        assert mock_conv == pytest.approx(0.062, 1e-2)
        assert mock_work == pytest.approx(12, 1e-2)
        assert mock_efficiency == pytest.approx(0.792, 1e-2)

    def test_create_uniform_coarsening_domain_optimize_kh_0_5_repetitive(self):
        n = 96
        kh = 0.5
        max_conv_factor = 0.3
        # Create fine-level matrix.
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = hm.setup.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10, num_examples=2)

        # Construct coarsening.
        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, 4, repetitive=True)
        r, aggregate_size, nc, cr, mean_energy_error, mock_conv, mock_work, mock_efficiency = \
            coarsener.get_optimal_coarsening(max_conv_factor)

        assert r.shape == (48, 96)
        assert aggregate_size == 4
        assert nc == 2
        assert cr == pytest.approx(0.5, 1e-2)
        assert mean_energy_error == pytest.approx(0.127, 1e-2)
        assert mock_conv == pytest.approx(0.152, 1e-2)
        assert mock_work == pytest.approx(8, 1e-2)
        assert mock_efficiency == pytest.approx(0.789, 1e-2)


def _get_test_matrix(a, n, num_sweeps, num_examples: int = None):
    level = hm.setup.hierarchy.create_finest_level(a)
    x = hm.solve.run.random_test_matrix((n,), num_examples=num_examples)
    b = np.zeros_like(x)
    x, _ = hm.solve.run.run_iterative_method(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)
    return x
