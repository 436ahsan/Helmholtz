"""Tests AutoAMG bootstrap coarsening for a linear problem A*x=b: 5-point Helmholtz discretization with periodic
boundary conditions."""
import logging
import sys
import numpy as np
import pytest
from numpy.ma.testutils import assert_array_equal, assert_array_almost_equal
from scipy.linalg import norm

import helmholtz as hm
import helmholtz.setup.hierarchy as hierarchy

logger = logging.getLogger("nb")


class TestBootstrapAuto:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.set_printoptions(precision=6, linewidth=1000)
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(stream=sys.stdout, level=logging.WARN, format="%(message)s")
        np.random.seed(0)
        np.set_printoptions(linewidth=200, precision=2)

    def test_run_1_level_relax(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_5_point_operator(kh, n)
        level = hierarchy.create_finest_level(a)
        multilevel = hm.hierarchy.multilevel.Multilevel(level)
        x = hm.solve.run.random_test_matrix((n,), num_examples=1)
        multilevel = hm.hierarchy.multilevel.Multilevel(level)
        # Run enough Kaczmarz relaxations per lambda update (not just 1 relaxation) so we converge to the minimal one.
        nu = 1
        method = lambda x: hm.solve.relax_cycle.relax_cycle(multilevel, 1.0, None, None, nu).run(x)
        x, conv_factor = hm.solve.run.run_iterative_method(level.operator, method, x, 100)

        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.1001, 1e-3)
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.114, 1e-2)
        assert conv_factor == pytest.approx(0.99979, 1e-2)

    def test_laplace_coarsening(self):
        n = 16
        kh = 0
        a = hm.linalg.helmholtz_1d_5_point_operator(kh, n).tocsr()

        x, multilevel = hm.setup.auto_setup.setup(a, max_levels=2, num_examples=20, num_sweeps=20,
                                                         threshold=0.2)

        assert x.shape == (16, 20)

        assert len(multilevel) == 2

        level = multilevel.finest_level
        assert level.a.shape == (16, 16)

        coarse_level = multilevel.level[1]
        assert coarse_level.a.shape == (8, 8)
        assert coarse_level._r_csr.shape == (8, 16)
        assert coarse_level._p_csr.shape == (16, 8)
        coarse_level.print()

        # Test that test vector residuals are small.
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.120, 1e-2)

        # Test two-level cycle convergence for A*x=0.
        two_level_cycle = lambda x: hm.solve.solve_cycle.solve_cycle(multilevel, 1.0, 1, 1).run(x)
        x0 = np.random.random((a.shape[0], ))
        x, conv_factor = hm.solve.run.run_iterative_method(level.operator, two_level_cycle, x0, 20)
        assert conv_factor == pytest.approx(0.23, 1e-2)


    def test_laplace_2_level_bootstrap(self):
        """We improve vectors by relaxation -> coarsening creation -> 2-level relaxation cycles.
        P = SVD interpolation = R^T."""
        n = 16
        kh = 0
        a = hm.linalg.helmholtz_1d_5_point_operator(kh, n).tocsr()

        x, multilevel = hm.setup.auto_setup.setup(a, max_levels=2, num_examples=20, num_sweeps=20,
                                                         threshold=0.2, num_bootstrap_steps=2)

        assert x.shape == (16, 20)
        assert len(multilevel) == 2

        # The coarse level should be Galerkin coarsening with piecewise constant interpolation.
        coarse_level = multilevel.level[1]

        # Interpolation ~ 2nd order interpolation from 3 to 4 coarse neighbors.
        p = coarse_level.p
        assert_array_equal(p[0].nonzero()[0], [0, 0, 0])
        assert_array_equal(p[0].nonzero()[1], [0, 1, 7])
        assert_array_almost_equal(p[0].data, [-0.73, 0.13, -0.11], decimal=2)
        # Row sums are ~ constant.
        assert np.array(p.sum(axis=1)).flatten().mean() == pytest.approx(-0.707, 1e-2)
        assert np.array(p.sum(axis=1)).flatten().std() < 1e-4

        # Coarsening ~ averaging.
        r = coarse_level.r
        assert_array_equal(r[0].nonzero()[0], [0, 0])
        assert_array_equal(r[0].nonzero()[1], [0, 1])
        assert_array_almost_equal(r[0].data, [-0.7, -0.71], decimal=2)

        ac_0 = coarse_level.a[0]
        coarse_level.print()
        assert_array_equal(ac_0.nonzero()[1], [0, 1, 2, 3, 5, 6, 7])
        assert_array_almost_equal(ac_0.data,
                                  [-0.941028,  0.564876, -0.107981,  0.014992,  0.013662, -0.112899,  0.568408],
                                  decimal=5)

        # Vectors have lower residual after 2-level relaxation cycles than after relaxation only.
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.0315, 1e-2)

    def test_helmholtz_coarsening(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_5_point_operator(kh, n).tocsr()

        x, multilevel = hm.setup.auto_setup.setup(a, max_levels=2, num_examples=20, num_sweeps=20,
                                                         threshold=0.2)

        assert x.shape == (16, 20)

        assert len(multilevel) == 2

        level = multilevel.finest_level
        assert level.a.shape == (16, 16)

        coarse_level = multilevel.level[1]
        assert coarse_level.a.shape == (8, 8)
        assert coarse_level._r_csr.shape == (8, 16)
        assert coarse_level._p_csr.shape == (16, 8)
        coarse_level.print()

        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.145, 1e-2)

    def test_helmholtz_2_level_bootstrap_cycles_reduce_test_function_residual(self):
        """We improve vectors by relaxation -> coarsening creation -> 2-level relaxation cycles.
        P = SVD interpolation = R^T."""
        # Larger domain, as in Karsten's experiments.
        n = 96
        kh = 0.5
        num_examples = 20
        max_levels = 2

        # Initialize test functions (to random) and hierarchy at coarsest level.
        a = hm.linalg.helmholtz_1d_5_point_operator(kh, n).tocsr()
        level = hierarchy.create_finest_level(a)
        multilevel = hm.hierarchy.multilevel.Multilevel(level)
        x = hm.solve.run.random_test_matrix((a.shape[0],), num_examples=num_examples)
        assert norm(a.dot(x)) / norm(x) == pytest.approx(2.91, 1e-2)

        # Smooth TF by relaxation.
        b = np.zeros_like(x)
        x, relax_conv_factor = hm.solve.run.run_iterative_method(
            level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)
        assert norm(a.dot(x)) / norm(x) == pytest.approx(0.08, 1e-2)

        # Residual norm decreases fast during the first 3 bootstrap cycles, then saturates.
        expected_residual_norms = [0.175, 0.0677, 0.0502, 0.0473, 0.0444, 0.044, 0.0417, 0.0454]
        expected_conv_factor = [0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36]

        # Relax vector + coarsen in first iteration; then 2-level cycle + improve hierarchy (bootstrap).
        for i, expected_residual_norm in enumerate(expected_residual_norms):
            x, multilevel = hm.setup.auto_setup.bootstap(x, multilevel, max_levels, relax_conv_factor, num_sweeps=10)
            assert norm(a.dot(x)) / norm(x) == pytest.approx(expected_residual_norm, 1e-2)

            # Test two-level cycle convergence for A*x=0.
            two_level_cycle = lambda x: hm.solve.solve_cycle.solve_cycle(multilevel, 1.0, 1, 1).run(x)
            x0 = np.random.random((a.shape[0], ))
            _, conv_factor = hm.solve.run.run_iterative_method(level.operator, two_level_cycle, x0, 20)
            assert conv_factor == pytest.approx(expected_conv_factor[i], 1e-2)

    # def test_2_level_bootstrap_least_squares_interpolation(self):
    #     n = 16
    #     kh = 0.5
    #     a = hm.linalg.helmholtz_1d_operator(kh, n)
    #     x, multilevel = hm.repetitive.bootstrap_repetitive._repetitive_eigen.generate_test_matrix(a, 0, num_examples=4, interpolation_method="ls")
    #     assert len(multilevel) == 2
    #
    #     level = multilevel.finest_level
    #     # Convergence speed test.
    #     relax_cycle = lambda x: hm.setup_eigen.eigensolver.relax_cycle(multilevel, 1.0, 1, 1, 100).run(x)
    #     # FMG start so x has a reasonable initial guess.
    #     x = hm.repetitive.bootstrap_repetitive._repetitive_eigen.fmg(multilevel, num_cycles_finest=0)
    #     x, conv_factor = hm.solve.run.run_iterative_eigen_method(level.operator, relax_cycle, x, 20, print_frequency=1)
    #
    #     assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
    #     assert conv_factor == pytest.approx(0.316, 1e-2)
    #
    # #@unittest.skip("3-level not working well, solve 2-level well enough first.")
    # def test_3_level_fixed_domain(self):
    #     n = 16
    #     kh = 0.5
    #     a = hm.linalg.helmholtz_1d_operator(kh, n)
    #     x, multilevel = hm.repetitive.bootstrap_repetitive._repetitive_eigen.generate_test_matrix(
    #         a, 0, num_sweeps=20, num_examples=4, initial_max_levels=3)
    #     assert len(multilevel) == 3
    #
    #     level = multilevel.finest_level
    #
    #     # Convergence speed test.
    #     # FMG start so x has a reasonable initial guess.
    #     x_init = hm.repetitive.bootstrap_repetitive._repetitive_eigen.fmg(multilevel, num_cycles_finest=0, num_cycles=1)
    #     #        multilevel.lam = exact_eigenpair(level.a)
    #
    #     relax_cycle = lambda x: hm.setup_eigen.eigensolver.relax_cycle(multilevel, 1.0, 1, 1, 100, num_levels=3).run(x)
    #     x, conv_factor = hm.solve.run.run_iterative_eigen_method(level.operator, relax_cycle, x_init, 15)
    #     assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
    #     assert conv_factor == pytest.approx(0.32, 1e-2)
