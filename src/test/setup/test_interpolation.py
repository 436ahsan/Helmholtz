import logging
import unittest

import numpy as np
import pytest
from numpy.ma.testutils import assert_array_equal, assert_array_almost_equal
from numpy.linalg import norm

import helmholtz as hm
import helmholtz.analysis
import helmholtz.repetitive.coarsening_repetitive as cr


class TestInterpolation:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.random.seed(0)
        hm.logging.set_simple_logging(logging.WARN)

    def test_create_coarsening_domain_non_repetitive(self):
        n = 32
        kh = 0.6
        num_sweeps = 100
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        location = np.arange(n)
        # Generate relaxed test matrix.
        level = hm.setup.hierarchy.create_finest_level(a)
        x = hm.solve.run.random_test_matrix((n,))
        b = np.zeros_like(x)
        x, _ = hm.solve.run.run_iterative_method(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)
        assert x.shape == (32, 128)
        # Generate coarse variables (R) on the non-repetitive domain. Use a uniform coarsening.
        r, aggregates, num_components, energy_error = cr.create_coarsening_domain(x, threshold=0.15)

        p = hm.setup.interpolation.create_interpolation_least_squares_domain(x, a, r, location, n, target_error=0.07)

        num_test_examples = 5
        x_test = x[:, -num_test_examples:]
        error_a = np.mean(norm(a.dot(x_test - p.dot(r.dot(x_test))), axis=0) / norm(x_test, axis=0))
        assert p[0].nnz == 3
        assert error_a == pytest.approx(0.0659, 1e-2)
        assert p.shape == (32, 16)
        # To print p:
        print(','.join(np.array2string(y, separator=",", formatter={'float_kind':lambda x: "%.2f" % x}) for y in np.array(p.todense())))
        assert_array_almost_equal(p.todense(), [
            [0.44, -0.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.05, 0.00], [0.60, -0.23, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.58, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.41, 0.63, -0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.23, 0.63, -0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, -0.56, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, -0.61, -0.19, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, -0.48, -0.57, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, -0.31, -0.58, 0.26, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.57, -0.28, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.60, 0.21, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.46, 0.58, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.28, 0.56, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.60, 0.21, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.57, -0.28, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.38, -0.65, -0.03, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.19, -0.64, -0.23, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.59, -0.24, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.59, 0.26, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.42, 0.62, 0.05, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.24, 0.58, 0.27, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.60, -0.23, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.58, 0.27, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40, 0.63, -0.06, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.23, 0.59, -0.28, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.60, 0.24,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.58, -0.26,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.41, -0.62,
                           0.05, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.24, -0.59,
             0.28, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.60, -0.24],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.59, 0.25], [0.32, -0.52, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.35, 0.00]
        ], decimal=2)

    def test_create_interpolation_least_squares_domain_repetitive(self):
        n = 32
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        location = np.arange(n)
        level = hm.setup.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10, num_examples=4)
        max_conv_factor = 0.3
        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, 4, repetitive=True)
        r, aggregate_size, num_components = coarsener.get_optimal_coarsening(max_conv_factor)[:3]
        assert aggregate_size == 4
        assert num_components == 2

        p = hm.setup.interpolation.create_interpolation_least_squares_domain(
            x, a, r, location, n, aggregate_size=aggregate_size, num_components=num_components, repetitive=True, target_error=0.07)

        error_a = np.mean(norm(a.dot(x - p.dot(r.dot(x))), axis=0) / norm(x, axis=0))
        assert p[0].nnz == 4
        assert error_a == pytest.approx(0.153, 1e-2)
        assert p.shape == (32, 16)

    @unittest.skip("At the moment we do not support domain size indivisible by aggregate size")
    def test_create_interpolation_least_squares_domain_repetitive_indivisible_size(self):
        n = 33
        kh = 0.6
        max_conv_factor = 0.3
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        location = np.arange(n)
        level = hm.setup.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10, num_examples=4)
        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, 4, repetitive=True)
        r, aggregate_size, num_components = coarsener.get_optimal_coarsening(max_conv_factor)[:3]
        assert aggregate_size == 4
        assert num_components == 2

        p = hm.setup.interpolation.create_interpolation_least_squares_domain(
            x, a, r, location,aggregate_size=aggregate_size, num_components=num_components, repetitive=True, target_error=0.04)

        num_test_examples = 5
        x_test = x[:, -num_test_examples:]
        error_a = np.mean(norm(a.dot(x_test - p.dot(r.dot(x_test))), axis=0) / norm(x_test, axis=0))
        assert p[0].nnz == 4
        assert error_a == pytest.approx(0.254, 1e-2)
        assert p.shape == (33, 18)

        # To print p:
        #print(','.join(np.array2string(y, separator=",", formatter={'float_kind':lambda x: "%.2f" % x}) for y in np.array(p.todense())))
        # assert_array_almost_equal(p.todense(), [
        #     [0.59, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.19], [0.57, 0.21, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        #     [0.19, 0.59, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], [0.00, 0.57, 0.21, 0.00, 0.00, 0.00, 0.00, 0.00],
        #     [0.00, 0.19, 0.59, 0.00, 0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.57, 0.21, 0.00, 0.00, 0.00, 0.00],
        #     [0.00, 0.00, 0.19, 0.59, 0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.57, 0.21, 0.00, 0.00, 0.00],
        #     [0.00, 0.00, 0.00, 0.19, 0.59, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.57, 0.21, 0.00, 0.00],
        #     [0.00, 0.00, 0.00, 0.00, 0.19, 0.59, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.57, 0.21, 0.00],
        #     [0.00, 0.00, 0.00, 0.00, 0.00, 0.19, 0.59, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.57, 0.21],
        #     [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.19, 0.59], [0.21, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.57],
        # ], decimal=2)

    def test_create_interpolation_least_squares_domain_repetitive_large_aggregate(self):
        n = 16
        kh = 1.0
        max_conv_factor = 0.3
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        location = np.arange(n)
        level = hm.setup.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10, num_examples=8)
        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, 4, repetitive=True)
        r, aggregate_size, num_components = coarsener.get_optimal_coarsening(max_conv_factor)[:3]
        assert aggregate_size == 6
        assert num_components == 2

        p = hm.setup.interpolation.create_interpolation_least_squares_domain(
            x, a, r, location,aggregate_size=aggregate_size, num_components=num_components, repetitive=True, target_error=0.07)

        num_test_examples = 5
        x_test = x[:, -num_test_examples:]
        error_a = np.mean(norm(a.dot(x_test - p.dot(r.dot(x_test))), axis=0) / norm(x_test, axis=0))
        assert p[0].nnz == 4
        assert error_a == pytest.approx(0.492, 1e-2)
        assert p.shape == (16, 6)

        # To print p:
        #print(','.join(np.array2string(y, separator=",", formatter={'float_kind':lambda x: "%.2f" % x}) for y in np.array(p.todense())))
        assert_array_almost_equal(p.todense(), [
            [-0.26,0.17,0.00,0.00,-0.20,0.14],[0.07,0.56,0.00,0.00,-0.08,0.20],[0.27,0.11,0.00,0.00,0.16,0.02],
            [0.34,-0.25,0.01,-0.02,0.00,0.00],[-0.03,-0.40,-0.13,-0.05,0.00,0.00],[-0.30,-0.16,-0.11,-0.01,0.00,0.00],
            [-0.20,0.14,-0.26,0.17,0.00,0.00],[-0.08,0.20,0.07,0.56,0.00,0.00],[0.16,0.02,0.27,0.11,0.00,0.00],
            [0.00,0.00,0.34,-0.25,0.01,-0.02],[0.00,0.00,-0.22,-0.26,-0.39,0.12],[0.00,0.00,-0.38,0.04,-0.04,0.55],
            [0.00,0.00,0.16,0.02,0.27,0.11],[0.01,-0.02,0.00,0.00,0.34,-0.25],[-0.13,-0.05,0.00,0.00,-0.03,-0.40],
            [-0.11,-0.01,0.00,0.00,-0.30,-0.16],
        ], decimal=2)

    def test_create_interpolation_least_squares_domain_ideal_tvs(self):
        n = 32
        kh = 0.6
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        location = np.arange(n)
        # Generate relaxed test matrix.
        x, _ = helmholtz.analysis.ideal.ideal_tv(a, 10)
        assert x.shape == (32, 10)
        # Generate coarse variables (R) on the non-repetitive domain.
        r, aggregates, num_components, energy_error = cr.create_coarsening_domain(x, threshold=0.15)

        aggregate_size = np.array([len(aggregate) for aggregate in aggregates])
        assert_array_equal(aggregate_size, [6, 6, 4, 6, 6, 4])
        assert_array_equal(num_components, [3, 3, 2, 3, 3, 2])

        # Force a small caliber since our sample size is small ((10 - some for testing) / 2 for fitting).
        p = hm.setup.interpolation.create_interpolation_least_squares_domain(
            x, a, r, location, n,  neighborhood="aggregate", caliber=2)

    def test_create_interpolation_least_squares_domain_repetitive_large_aggregate(self):
        n = 18
        kh = 1.0
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        location = np.arange(n)

        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 30, num_examples=8)

        # Generate coarsening.
        aggregate_size = 6
        num_components = 2
        coarsener = hm.setup.coarsening_uniform.FixedAggSizeUniformCoarsener(x, aggregate_size)
        r, mean_energy_error = coarsener[num_components]
        assert mean_energy_error == pytest.approx(0.0128, 1e-2)

        p = hm.setup.interpolation.create_interpolation_least_squares_domain(
            x, a, r, location, n, aggregate_size=aggregate_size, num_components=num_components, repetitive=True)

        num_test_examples = 5
        x_test = x[:, -num_test_examples:]
        error_l2 = np.mean(norm(x_test - p.dot(r.dot(x_test)), axis=0) / norm(x_test, axis=0))
        error_a = np.mean(norm(a.dot(x_test - p.dot(r.dot(x_test))), axis=0) / norm(x_test, axis=0))
        assert p[0].nnz == 2
        assert error_l2 == pytest.approx(0.0236, 1e-2)
        assert error_a == pytest.approx(0.035, 1e-2)
        assert p.shape == (18, 6)

        # To print p:
        #print(','.join(np.array2string(y, separator=",", formatter={'float_kind':lambda x: "%.2f" % x}) for y in np.array(p.todense())))
        assert_array_almost_equal(p.todense(), [
            [-0.23,-0.53,0.00,0.00,0.00,0.00],[-0.57,-0.06,0.00,0.00,0.00,0.00],[-0.34,0.47,0.00,0.00,0.00,0.00],[0.23,0.53,0.00,0.00,0.00,0.00],[0.57,0.06,0.00,0.00,0.00,0.00],[0.34,-0.47,0.00,0.00,0.00,0.00],[0.00,0.00,-0.23,-0.53,0.00,0.00],[0.00,0.00,-0.57,-0.06,0.00,0.00],[0.00,0.00,-0.34,0.47,0.00,0.00],[0.00,0.00,0.23,0.53,0.00,0.00],[0.00,0.00,0.57,0.06,0.00,0.00],[0.00,0.00,0.34,-0.47,0.00,0.00],[0.00,0.00,0.00,0.00,-0.23,-0.53],[0.00,0.00,0.00,0.00,-0.57,-0.06],[0.00,0.00,0.00,0.00,-0.34,0.47],[0.00,0.00,0.00,0.00,0.23,0.53],[0.00,0.00,0.00,0.00,0.57,0.06],[0.00,0.00,0.00,0.00,0.34,-0.47]
        ], decimal=2)

    @unittest.skip("WIP")
    def test_create_interpolation_least_squares_domain_repetitive_extrapolation(self):
        ""
        n = 16
        kh = 1.0
        max_conv_factor = 0.3
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        location = np.arange(n)
        level = hm.setup.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10, num_examples=8)
        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, 4, repetitive=True)
        r, aggregate_size, num_components = coarsener.get_optimal_coarsening(max_conv_factor)[:3]
        assert aggregate_size == 6
        assert num_components == 2

        p = hm.setup.interpolation.create_interpolation_least_squares_domain(
            x, a, r, location, n, aggregate_size=aggregate_size, num_components=num_components, repetitive=True)

        num_test_examples = 5
        x_test = x[:, -num_test_examples:]
        error_a = np.mean(norm(a.dot(x_test - p.dot(r.dot(x_test))), axis=0) / norm(x_test, axis=0))
        assert p[0].nnz == 4
        assert error_a == pytest.approx(0.492, 1e-2)
        assert p.shape == (16, 6)

        # To print p:
        #print(','.join(np.array2string(y, separator=",", formatter={'float_kind':lambda x: "%.2f" % x}) for y in np.array(p.todense())))
        assert_array_almost_equal(p.todense(), [
            [-0.26,0.17,0.00,0.00,-0.20,0.14],[0.07,0.56,0.00,0.00,-0.08,0.20],[0.27,0.11,0.00,0.00,0.16,0.02],
            [0.34,-0.25,0.01,-0.02,0.00,0.00],[-0.03,-0.40,-0.13,-0.05,0.00,0.00],[-0.30,-0.16,-0.11,-0.01,0.00,0.00],
            [-0.20,0.14,-0.26,0.17,0.00,0.00],[-0.08,0.20,0.07,0.56,0.00,0.00],[0.16,0.02,0.27,0.11,0.00,0.00],
            [0.00,0.00,0.34,-0.25,0.01,-0.02],[0.00,0.00,-0.22,-0.26,-0.39,0.12],[0.00,0.00,-0.38,0.04,-0.04,0.55],
            [0.00,0.00,0.16,0.02,0.27,0.11],[0.01,-0.02,0.00,0.00,0.34,-0.25],[-0.13,-0.05,0.00,0.00,-0.03,-0.40],
            [-0.11,-0.01,0.00,0.00,-0.30,-0.16],
        ], decimal=2)

    def test_create_interpolation_least_squares_domain_repetitive_plain_weighted(self):
        """Weighted least-squares interpolation fitting."""
        n = 32
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        location = np.arange(n)
        level = hm.setup.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10, num_examples=4)
        max_conv_factor = 0.3
        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, 4, repetitive=True)
        r, aggregate_size, num_components = coarsener.get_optimal_coarsening(max_conv_factor)[:3]
        assert aggregate_size == 4
        assert num_components == 2

        p = hm.setup.interpolation.create_interpolation_least_squares_domain(
            x, a, r, location, n, aggregate_size=aggregate_size, num_components=num_components, repetitive=True, target_error=0.07,
            fit_scheme="plain", weighted=True)

        error_a = np.mean(norm(a.dot(x - p.dot(r.dot(x))), axis=0) / norm(x, axis=0))
        assert p[0].nnz == 4
        assert error_a == pytest.approx(0.128, 1e-2)
        assert p.shape == (32, 16)

    def test_create_interpolation_least_squares_domain_repetitive_ridge_weighted(self):
        """Weighted least-squares interpolation fitting."""
        n = 32
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        location = np.arange(n)
        level = hm.setup.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10, num_examples=4)
        max_conv_factor = 0.3
        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, 4, repetitive=True)
        r, aggregate_size, num_components = coarsener.get_optimal_coarsening(max_conv_factor)[:3]
        assert aggregate_size == 4
        assert num_components == 2

        p = hm.setup.interpolation.create_interpolation_least_squares_domain(
            x, a, r, location, n, aggregate_size=aggregate_size, num_components=num_components, repetitive=True, target_error=0.07,
            fit_scheme="ridge", weighted=True)

        error_a = np.mean(norm(a.dot(x - p.dot(r.dot(x))), axis=0) / norm(x, axis=0))
        assert p[0].nnz == 4
        assert error_a == pytest.approx(0.158, 1e-2)
        assert p.shape == (32, 16)


def _get_test_matrix(a, n, num_sweeps, num_examples: int = None):
    level = hm.setup.hierarchy.create_finest_level(a)
    x = hm.solve.run.random_test_matrix((n,), num_examples=num_examples)
    b = np.zeros_like(x)
    x, _ = hm.solve.run.run_iterative_method(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)
    return x
