import numpy as np
import pytest
from numpy.ma.testutils import assert_array_equal, assert_array_almost_equal

import helmholtz as hm
import helmholtz.analysis


class TestInterpolation:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.random.seed(0)

    def test_create_interpolation_least_squares_auto_nbhrs(self):
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
        r, aggregates, nc, energy_error = hm.setup.coarsening.create_coarsening_full_domain(x, threshold=0.15)

        p, fit_error, val_error, test_error, alpha_opt = \
            hm.setup.interpolation.create_interpolation_least_squares_auto_nbhrs(x, a, r)

        assert np.mean(alpha_opt) == pytest.approx(0.0015625)
        assert max(fit_error) == pytest.approx(0.0286, 1e-2)
        assert max(val_error) == pytest.approx(0.0355, 1e-2)
        assert max(test_error) == pytest.approx(0.0350, 1e-2)

        assert p.shape == (32, 16)
        # To print p:
        print(','.join(np.array2string(y, separator=",", formatter={'float_kind':lambda x: "%.2f" % x}) for y in np.array(p.todense())))
        assert_array_almost_equal(p.todense(), [
            [0.33, -0.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.07, 0.25], [0.60, -0.23, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.58, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.35, 0.45, -0.08, 0.19, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.08, 0.18, -0.32, 0.48, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, -0.56, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, -0.61, -0.19, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, -0.40, -0.41, 0.08, -0.19, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, -0.10, -0.16, 0.34, -0.48, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.57, -0.28, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.60, 0.21, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.39, 0.41, 0.10, 0.18, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.10, 0.16, 0.39, 0.44, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.60, 0.21, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.57, -0.28, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.34, -0.47, -0.09, -0.18, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.08, -0.18, -0.37, -0.44, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.59, -0.24, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.58, 0.25, 0.00, 0.00, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.36, 0.45, 0.10, -0.18, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.09, 0.18, 0.38, -0.44, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.60, -0.23, 0.00, 0.00,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.58, 0.27, 0.00, 0.00,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.35, 0.46, -0.10, 0.18,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.08, 0.18, -0.37, 0.45,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.59, 0.24,
                           0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.58, -0.26,
             0.00, 0.00], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.35, -0.46,
                           0.09, -0.18],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.08, -0.19,
             0.37, -0.44], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                            0.59, -0.24],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
             0.58, 0.25], [0.14, -0.18, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                           0.39, 0.42]
        ], decimal=2)

    def test_create_interpolation_least_squares_auto_nbhrs_ideal_tvs(self):
        n = 32
        kh = 0.6
        a = hm.linalg.helmholtz_1d_operator(kh, n).tocsr()
        # Generate relaxed test matrix.
        x, _ = helmholtz.analysis.ideal.ideal_tv(a, 10)
        assert x.shape == (32, 10)
        # Generate coarse variables (R) on the non-repetitive domain.
        r, aggregates, nc, energy_error = hm.setup.coarsening.create_coarsening_full_domain(x, threshold=0.15)

        aggregate_size = np.array([len(aggregate) for aggregate in aggregates])
        assert_array_equal(aggregate_size, [8, 4, 4, 8, 4, 4])
        assert_array_equal(nc, [4, 2, 2, 4, 2, 2])

        p, fit_error, val_error, test_error, alpha_opt = \
            hm.setup.interpolation.create_interpolation_least_squares_auto_nbhrs(x, a, r, neighborhood="aggregate")
