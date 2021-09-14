import logging
import numpy as np
import pytest
from numpy.ma.testutils import assert_array_equal, assert_array_almost_equal

import helmholtz as hm
import helmholtz.analysis


class TestInterpolation:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.random.seed(0)
        hm.logging.set_simple_logging(logging.DEBUG)

    def test_create_interpolation_least_squares_domain(self):
        n = 32
        kh = 0.6
        num_sweeps = 100
        a = hm.linalg.helmholtz_1d_operator(kh, n).tocsr()
        # Generate relaxed test matrix.
        level = hm.setup.hierarchy.create_finest_level(a)
        x = hm.solve.run.random_test_matrix((n,))
        b = np.zeros_like(x)
        x, _ = hm.solve.run.run_iterative_method(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)
        assert x.shape == (32, 128)
        # Generate coarse variables (R) on the non-repetitive domain.
        r, aggregates, nc, energy_error = hm.setup.coarsening.create_coarsening_domain(x, threshold=0.15)

        p = hm.setup.interpolation.create_interpolation_least_squares_domain(x, a, r)

        num_test_examples = 5
        x_test = x[-num_test_examples:]
        error_a = np.mean(norm(a.dot(x_test - p.dot(r.dot(x_test))), axis=0) / norm(x_test, axis=0))
        assert p[0].nnz == 3
        assert error_a == pytest.approx(0.0658, 1e-2)
        assert p.shape == (32, 16)
        # To print p:
        print(','.join(np.array2string(y, separator=",", formatter={'float_kind':lambda x: "%.2f" % x}) for y in np.array(p.todense())))
        assert_array_almost_equal(p.todense(), [
            [0.44, -0.61, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.04, 0.00], [0.60, -0.23, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.58, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.40, 0.64, -0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.20, 0.65, -0.23, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, -0.56, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, -0.61, -0.19, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, -0.49, -0.56, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, -0.30, -0.57, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.57, -0.28, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.60, 0.21, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.47, 0.57, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.29, 0.53, 0.29, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.60, 0.21, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.57, -0.29, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.39, -0.63, -0.05, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, -0.62, -0.25, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.59, -0.25, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.59, 0.25, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.41, 0.63, 0.04, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.22, 0.61, 0.25, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.60, -0.23, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.58, 0.27, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.40, 0.63, -0.05, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.23, 0.58, -0.28, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.59, 0.24,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.58, -0.26,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.42, -0.61,
                           0.06, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.25, -0.56,
             0.30, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.59, -0.24],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.59, 0.25], [0.32, -0.52, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.35, 0.00]
        ], decimal=2)

    def test_create_interpolation_least_squares_domain_repetitive(self):
        n = 32
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n).tocsr()
        level = hm.repetitive.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10, num_examples=4)
        max_conv_factor = 0.3
        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, 4, repetitive=True)
        r, aggregate_size, nc = coarsener.get_optimal_coarsening(max_conv_factor)[:3]
        assert aggregate_size == 4
        assert nc == 2

        p = hm.setup.interpolation.create_interpolation_least_squares_domain(
            x, a, r, aggregate_size=aggregate_size, nc=nc, repetitive=True)

        x_test = x[-num_test_examples:]
        error_a = np.mean(norm(a.dot(x_test - p.dot(r.dot(x_test))), axis=0) / norm(x_test, axis=0))
        assert p[0].nnz == 3
        assert error_a == pytest.approx(0.11, 1e-2)
        assert p.shape == (32, 16)

    def test_create_interpolation_least_squares_domain_repetitive_indivisible_size(self):
        n = 33
        kh = 0.6
        a = hm.linalg.helmholtz_1d_operator(kh, n).tocsr()
        level = hm.repetitive.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10, num_examples=4)
        # Calculate mock cycle predicted efficiency.
        aggregate_size_values = np.array([2, 4, 6])
        max_conv_factor = 0.3
        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, aggregate_size_values, 4, repetitive=True)
        r, aggregate_size, nc = coarsener.get_optimal_coarsening(max_conv_factor)[:3]
        assert aggregate_size == 6
        assert nc == 3

        p, fit_error, val_error, test_error, alpha_opt = \
            hm.setup.interpolation.create_interpolation_least_squares_domain(x, a, r, aggregate_size=aggregate_size,
                                                                             nc=nc, repetitive=True)

        assert np.mean(alpha_opt) == pytest.approx(0.0167, 1e-2)
        assert max(fit_error) == pytest.approx(0.123, 1e-2)
        assert max(val_error) == pytest.approx(0.138, 1e-2)
        # Invisible size currently causes large errors due to boundaries (?!).
        assert max(test_error) == pytest.approx(1.52, 1e-2)
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
        a = hm.linalg.helmholtz_1d_operator(kh, n).tocsr()
        level = hm.repetitive.hierarchy.create_finest_level(a)
        # Generate relaxed test matrix.
        x = _get_test_matrix(a, n, 10, num_examples=8)
        # Calculate mock cycle predicted efficiency.
        aggregate_size_values = np.array([2, 4, 6])
        max_conv_factor = 0.3
        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(level, x, aggregate_size_values, 4, repetitive=True)
        r, aggregate_size, nc = coarsener.get_optimal_coarsening(max_conv_factor)[:3]
        assert aggregate_size == 6
        assert nc == 2

        p, fit_error, val_error, test_error, alpha_opt = \
            hm.setup.interpolation.create_interpolation_least_squares_domain(x, a, r, aggregate_size=aggregate_size,
                                                                             nc=nc, repetitive=True)

        assert max(fit_error) == pytest.approx(0.347, 1e-2)
        assert max(val_error) == pytest.approx(0.119, 1e-2)
        assert max(test_error) == pytest.approx(0.31, 1e-2)
        assert p.shape == (16, 8)

        # To print p:
        #print(','.join(np.array2string(y, separator=",", formatter={'float_kind':lambda x: "%.2f" % x}) for y in np.array(p.todense())))
        assert_array_almost_equal(p.todense(), [
            [-0.36, 0.18, 0.00, 0.00, 0.00, 0.00, -0.02, -0.32], [0.05, 0.79, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.63, 0.42, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], [0.45, -0.13, -0.28, -0.09, 0.00, 0.00, 0.00, 0.00],
            [-0.02, -0.32, -0.36, 0.18, 0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.05, 0.79, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.63, 0.42, 0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.45, -0.13, -0.28, -0.09, 0.00, 0.00],
            [0.00, 0.00, -0.02, -0.32, -0.36, 0.18, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.05, 0.79, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.63, 0.42, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.45, -0.13, -0.28, -0.09],
            [0.00, 0.00, 0.00, 0.00, -0.02, -0.32, -0.36, 0.18], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.79],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.63, 0.42], [-0.28, -0.09, 0.00, 0.00, 0.00, 0.00, 0.45, -0.13]
        ], decimal=2)

    def test_create_interpolation_least_squares_domain_ideal_tvs(self):
        n = 32
        kh = 0.6
        a = hm.linalg.helmholtz_1d_operator(kh, n).tocsr()
        # Generate relaxed test matrix.
        x, _ = helmholtz.analysis.ideal.ideal_tv(a, 10)
        assert x.shape == (32, 10)
        # Generate coarse variables (R) on the non-repetitive domain.
        r, aggregates, nc, energy_error = hm.setup.coarsening.create_coarsening_domain(x, threshold=0.15)

        aggregate_size = np.array([len(aggregate) for aggregate in aggregates])
        assert_array_equal(aggregate_size, [6, 6, 4, 6, 6, 4])
        assert_array_equal(nc,[3, 3, 2, 3, 3, 2])

        p, fit_error, val_error, test_error, alpha_opt = \
            hm.setup.interpolation.create_interpolation_least_squares_domain(x, a, r, neighborhood="aggregate")


def _get_test_matrix(a, n, num_sweeps, num_examples: int = None):
    level = hm.repetitive.hierarchy.create_finest_level(a)
    x = hm.solve.run.random_test_matrix((n,), num_examples=num_examples)
    b = np.zeros_like(x)
    x, _ = hm.solve.run.run_iterative_method(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)
    return x
