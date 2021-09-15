import logging
import sys
import numpy as np
import pytest
from numpy.ma.testutils import assert_array_equal, assert_array_almost_equal
from scipy.linalg import norm

import helmholtz as hm

logger = logging.getLogger("nb")


class TestBootstrap5Point:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.set_printoptions(precision=6, linewidth=1000)
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(stream=sys.stdout, level=logging.WARN, format="%(message)s")
        np.random.seed(0)

    def test_run_1_level_relax(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_5_point_operator(kh, n)
        level = old_code.repetitive.hierarchy_repetitive.create_finest_level(a)
        multilevel = hm.hierarchy.multilevel.Multilevel.create(level)
        x = hm.solve.run.random_test_matrix((n,), num_examples=1)
        multilevel = hm.hierarchy.multilevel.Multilevel.create(level)
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

        a = hm.linalg.helmholtz_1d_5_point_operator(kh, n)
        x, multilevel = old_code.repetitive.bootstrap_repetitive.generate_test_matrix(a, 0, num_examples=4, num_sweeps=20)

        assert x.shape == (16, 4)

        assert len(multilevel) == 2

        level = multilevel.finest_level
        assert level.a.shape == (16, 16)

        coarse_level = multilevel[1]
        assert coarse_level.a.shape == (8, 8)
        assert coarse_level._r_csr.shape == (8, 16)
        assert coarse_level._p_csr.shape == (16, 8)
        coarse_level.print()

        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.222, 1e-2)

    def test_laplace_2_level_bootstrap(self):
        """We improve vectors by relaxation -> coarsening creation -> 2-level relaxation cycles.
        P = SVD interpolation = R^T."""
        n = 16
        kh = 0

        a = hm.linalg.helmholtz_1d_5_point_operator(kh, n)
        x, multilevel = old_code.repetitive.bootstrap_repetitive.generate_test_matrix(
            a, 0, num_examples=4, num_sweeps=20, num_bootstrap_steps=2)

        assert x.shape == (16, 4)
        assert len(multilevel) == 2

        # The coarse level should be Galerkin coarsening with piecewise constant interpolation.
        coarse_level = multilevel[1]

        p = coarse_level.p.asarray()
        assert_array_equal(p[0], [[0, 1], [0, 1], [0, 1], [0, 1]])
        assert_array_almost_equal(p[1], [[-0.458951, -0.704096],
             [-0.48365 , -0.25242 ],
             [-0.512496,  0.200809],
             [-0.541105,  0.632621]])

        r = coarse_level.r.asarray()
        assert_array_almost_equal(r, [[-0.458951, -0.48365 , -0.512496, -0.541105],
              [-0.704096, -0.25242 ,  0.200809,  0.632621]])

        ac_0 = coarse_level.a[0]
        coarse_level.print()
        assert_array_equal(ac_0.nonzero()[1], [0, 1, 2, 3, 6, 7])
        assert_array_almost_equal(ac_0.data, [-0.590408,  0.066061,  0.289712,  0.466534,  0.289712, -0.353946])

        # Vectors have much lower residual after 2-level relaxation cycles.
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.0859, 1e-3)

    def test_helmholtz_coarsening(self):
        n = 16
        kh = 0.5

        a = hm.linalg.helmholtz_1d_5_point_operator(kh, n)
        x, multilevel = old_code.repetitive.bootstrap_repetitive.generate_test_matrix(a, 0, num_examples=4, num_sweeps=20)

        assert x.shape == (16, 4)

        assert len(multilevel) == 2

        level = multilevel.finest_level
        assert level.a.shape == (16, 16)

        coarse_level = multilevel[1]
        assert coarse_level.a.shape == (8, 8)
        assert coarse_level._r_csr.shape == (8, 16)
        assert coarse_level._p_csr.shape == (16, 8)
        coarse_level.print()

        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.226, 1e-2)

    def test_helmholtz_2_level_bootstrap(self):
        """We improve vectors by relaxation -> coarsening creation -> 2-level relaxation cycles.
        P = SVD interpolation = R^T."""
        # Larger domain, as in Karsten's experiments.
        n = 96
        kh = 0.5
        num_examples = 1
        max_levels = 2

        # Initialize test functions (to random) and hierarchy at coarsest level.
        a = hm.linalg.helmholtz_1d_5_point_operator(kh, n)
        level = old_code.repetitive.hierarchy_repetitive.create_finest_level(a)
        multilevel = hm.hierarchy.multilevel.Multilevel.create(level)
        domain_shape = (a.shape[0],)
        x = hm.solve.run.random_test_matrix(domain_shape, num_examples=num_examples)
        assert norm(a.dot(x)) / norm(x) == pytest.approx(3.155, 1e-3)

        # Residual norm decreases fast during the first 3 bootstrap cycles, then saturates.
        expected_residual_norms = [0.276, 0.2144, 0.0901, 0.0626, 0.0509, 0.0464, 0.0442, 0.04278]

        # Relax vector + coarsen in first iteration; then 2-level cycle + improve hierarchy (bootstrap).
        for i, expected_residual_norm in enumerate(expected_residual_norms):
            x, multilevel = old_code.repetitive.bootstrap_repetitive.bootstap(x, multilevel, max_levels, num_sweeps=10)
            assert norm(a.dot(x)) / norm(x) == pytest.approx(expected_residual_norm, 1e-3)

    def test_helmholtz_2_level_more_bootstrap_doesnt_change_residual(self):
        """We improve vectors by relaxation -> coarsening creation -> 2-level relaxation cycles.
        P = SVD interpolation = R^T."""
        n = 16
        kh = 0.5

        a = hm.linalg.helmholtz_1d_5_point_operator(kh, n)
        x, multilevel = old_code.repetitive.bootstrap_repetitive.generate_test_matrix(a, 0, num_examples=4, num_sweeps=20, num_bootstrap_steps=3)

        assert x.shape == (16, 4)
        assert len(multilevel) == 2

        # Vectors have much lower residual after 2-level relaxation cycles.
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.1185, 1e-3)


    def test_2_level_bootstrap_least_squares_interpolation(self):
        """We improve vectors by relaxation -> coarsening creation -> 2-level relaxation cycles.
        P = SVD interpolation = R^T."""
        n = 16
        kh = 0.5

        a = hm.linalg.helmholtz_1d_5_point_operator(kh, n)
        x, multilevel = old_code.repetitive.bootstrap_repetitive.generate_test_matrix(a, 0, num_examples=10, num_sweeps=20, num_bootstrap_steps=1,
                                                                                      interpolation_method="ls")

        assert x.shape == (16, 10)
        assert len(multilevel) == 2

        # Vectors have much lower residual after 2-level relaxation cycles.
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.22, 1e-2)

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
