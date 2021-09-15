import logging
import sys
import numpy as np
import pytest
import unittest
from numpy.ma.testutils import assert_array_equal, assert_array_almost_equal

import helmholtz as hm

logger = logging.getLogger("nb")


class TestBootstrapRepetitive:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.set_printoptions(precision=6, linewidth=1000)
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(stream=sys.stdout, level=logging.WARN, format="%(message)s")
        np.random.seed(0)

    def test_run_1_level_relax(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = old_code.repetitive.hierarchy_repetitive.create_finest_level(a)
        multilevel = old_code.repetitive.hierarchy_repetitive.multilevel.Multilevel.create(level)
        x = hm.solve.run.random_test_matrix((n,), num_examples=1)
        multilevel = old_code.repetitive.hierarchy_repetitive.multilevel.Multilevel.create(level)
        # Run enough Kaczmarz relaxations per lambda update (not just 1 relaxation) so we converge to the minimal one.
        nu = 1
        method = lambda x: hm.solve.relax_cycle.relax_cycle(multilevel, 1.0, None, None, nu).run(x)
        x, conv_factor = hm.solve.run.run_iterative_method(level.operator, method, x, 100)

        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.0979, 1e-3)
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.110, 1e-2)
        assert conv_factor == pytest.approx(0.9957, 1e-3)

    def test_laplace_coarsening(self):
        n = 16
        kh = 0

        a = hm.linalg.helmholtz_1d_operator(kh, n)
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
               pytest.approx(0.178, 1e-2)

    def test_laplace_2_level_bootstrap(self):
        """We improve vectors by relaxation -> coarsening creation -> 2-level relaxation cycles.
        P = SVD interpolation = R^T."""
        n = 16
        kh = 0

        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = old_code.repetitive.bootstrap_repetitive.generate_test_matrix(a, 0, num_examples=4, num_sweeps=20, num_bootstrap_steps=2)

        assert x.shape == (16, 4)
        assert len(multilevel) == 2

        # The coarse level should be Galerkin coarsening with piecewise constant interpolation.
        coarse_level = multilevel[1]

        p = coarse_level.p.asarray()
        assert_array_equal(p[0], [[0], [0]])
        assert_array_almost_equal(p[1], [[-0.707228], [-0.706985]])

        r = coarse_level.r.asarray()
        assert_array_almost_equal(r, [[-0.707228, -0.706985]])

        ac_0 = coarse_level.a[0]
        coarse_level.print()
        assert_array_equal(ac_0.nonzero()[1], [0, 1, 7])
        assert_array_almost_equal(ac_0.data, [-1. ,  0.5,  0.5])

        # Vectors have much lower residual after 2-level relaxation cycles.
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.045054, 1e-3)

    def test_laplace_2_level_more_bootstrap_improves_vectors(self):
        n = 16
        kh = 0

        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = old_code.repetitive.bootstrap_repetitive.generate_test_matrix(a, 0, num_examples=4, num_sweeps=20, num_bootstrap_steps=3)

        assert x.shape == (16, 4)
        coarse_level = multilevel[1]
        coarse_level.print()

        # Vectors have much lower residual after 2-level relaxation cycles.
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.0010850, 1e-3)

    def test_2_level_one_bootstrap_step_improves_convergence(self):
        n = 48
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        # Threshold = 0.12 gives 2 SVD components
        x, multilevel = old_code.repetitive.bootstrap_repetitive.generate_test_matrix(a, 0, num_examples=4, threshold=0.12)
        assert len(multilevel) == 2

        level = multilevel.finest_level

        # Convergence speed test.
        relax_cycle = lambda x, lam: old_code.setup_eigen.eigensolver.eigen_cycle(
            multilevel, 1.0, 1, 1, 100).run((x, lam))
        # FMG start so x has a reasonable initial guess.
        x = old_code.repetitive.bootstrap_repetitive.fmg(multilevel, num_cycles_finest=0)
        x, lam, conv_factor = hm.solve.run.run_iterative_eigen_method(level.operator, relax_cycle, x, 20, print_frequency=1)

#        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.475, 1e-2)

    def test_2_level_two_bootstrap_steps_same_speed_as_one(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = old_code.repetitive.bootstrap_repetitive.generate_test_matrix(a, 0, num_examples=4, num_bootstrap_steps=2)
        assert len(multilevel) == 2

        level = multilevel.finest_level

        # Convergence speed test.
        relax_cycle = lambda x, lam: old_code.setup_eigen.eigensolver.eigen_cycle(
            multilevel, 1.0, 4, 3, 100).run((x, lam))
        # FMG start so x has a reasonable initial guess.
        logger.info("2-level convergence test")
        x = old_code.repetitive.bootstrap_repetitive.fmg(multilevel, num_cycles_finest=0)

        # Add some random noise but still stay near a reasonable initial guess.
        # x += 0.1 * np.random.random(x.shape)
        # multilevel.finest_multilevel.lam *= 1.01

        x, lam, conv_factor = hm.solve.run.run_iterative_eigen_method(level.operator, relax_cycle, x, 20,
                                                          print_frequency=1, residual_stop_value=1e-11)

#        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.156, 1e-2)

    def test_2_level_bootstrap_least_squares_interpolation_laplace(self):
        n = 16
        kh = 0
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = old_code.repetitive.bootstrap_repetitive.generate_test_matrix(
            a, 0, num_examples=4, interpolation_method="ls", num_sweeps=1000)
        assert len(multilevel) == 2

        # Convergence speed test.
        level = multilevel.finest_level
        eigen_cycle = lambda x, lam: old_code.setup_eigen.eigensolver.eigen_cycle(
            multilevel, 1.0, 1, 1, 100).run((x, lam))
        # FMG start so (x, lambda) has a reasonable initial guess.
        x = old_code.setup_eigen.bootstrap_eigen.fmg(multilevel, num_cycles_finest=0)
        lam = level.rq(x[:, 0])
        x, lam, conv_factor = hm.solve.run.run_iterative_eigen_method(level.operator, eigen_cycle, x, lam, 20, print_frequency=1)

#        assert lam == pytest.approx(0.0977590650225, 1e-3)
#        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.171, 1e-2)

    @unittest.skip("LS interpolation causes NaNs in this run")
    def test_2_level_bootstrap_least_squares_interpolation(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = old_code.repetitive.bootstrap_repetitive.generate_test_matrix(a, 0, num_examples=4, interpolation_method="ls")
        assert len(multilevel) == 2

        level = multilevel.finest_level
        # Convergence speed test.
        relax_cycle = lambda x, lam: old_code.setup_eigen.eigensolver.eigen_cycle(
            multilevel, 1.0, 1, 1, 100).run((x, lam))
        # FMG start so x has a reasonable initial guess.
        x = old_code.repetitive.bootstrap_repetitive.fmg(multilevel, num_cycles_finest=0)
        lam = level.rq(x[:, 0])
        x, lam, conv_factor = hm.solve.run.run_iterative_eigen_method(level.operator, relax_cycle, x, 20, print_frequency=1)

        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.316, 1e-2)

    @unittest.skip("3-level not working yet, can not find a good coarsening ratio.")
    def test_3_level_fixed_domain(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = old_code.repetitive.bootstrap_repetitive.generate_test_matrix(
            a, 0, num_sweeps=20, num_examples=4, initial_max_levels=3)
        assert len(multilevel) == 3

        level = multilevel.finest_level

        # Convergence speed test.
        # FMG start so x has a reasonable initial guess.
        x_init = old_code.repetitive.bootstrap_repetitivefmg(multilevel, num_cycles_finest=0, num_cycles=1)
        #        multilevel.lam = exact_eigenpair(level.a)

        relax_cycle = lambda x: old_code.setup_eigen.eigensolver.relax_cycle(multilevel, 1.0, 1, 1, 100, num_levels=3).run(x)
        x, conv_factor = hm.solve.run.run_iterative_eigen_method(level.operator, relax_cycle, x_init, 15)
        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.32, 1e-2)
