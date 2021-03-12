import logging
import sys
import numpy as np
import pytest
import unittest
from scipy.linalg import eig

import helmholtz as hm

logger = logging.getLogger("nb")


class TestBootstrap:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.set_printoptions(precision=6, linewidth=1000)
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(message)s")
        np.random.seed(0)

    def test_run_1_level_relax(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = hm.multilevel.Level.create_finest_level(a)
        multilevel = hm.multilevel.Multilevel(level)
        x = hm.run.random_test_matrix((n,), num_examples=1)
        multilevel = hm.multilevel.Multilevel(level)
        # Run enough Kaczmarz relaxations per lambda update (not just 1 relaxation) so we converge to the minimal one.
        nu = 1
        method = lambda x: hm.relax_cycle.relax_cycle(multilevel, 1.0, None, None, nu).run(x)
        x, conv_factor = hm.run.run_iterative_method(level.operator, method, x, 100)

        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.0979, 1e-3)
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.110, 1e-2)
        assert conv_factor == pytest.approx(0.9998, 1e-3)

    def test_1_level_bootstrap_laplace(self):
        n = 16
        kh = 0

        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a, 0, num_examples=4, num_sweeps=100)

        assert x.shape == (16, 4)

        assert len(multilevel) == 2

        level = multilevel.finest_level
        assert level.a.shape == (16, 16)

        coarse_level = multilevel.level[1]
        assert coarse_level.a.shape == (8, 8)
        assert coarse_level._r_csr.shape == (8, 16)
        assert coarse_level._p_csr.shape == (16, 8)
        coarse_level.print()

    def test_1_level_bootstrap_rays(self):
        n = 16
        kh = 0.5

        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a, 0, num_examples=4, num_sweeps=100)

        assert x.shape == (16, 4)

        assert len(multilevel) == 2

        level = multilevel.finest_level
        assert level.a.shape == (16, 16)

        coarse_level = multilevel.level[1]
        assert coarse_level.a.shape == (8, 8)
        assert coarse_level._r_csr.shape == (8, 16)
        assert coarse_level._p_csr.shape == (16, 8)
        coarse_level.print()

    def test_2_level_one_bootstrap_step_improves_convergence(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a, 0, num_examples=4)
        assert len(multilevel) == 2

        level = multilevel.finest_level
        # Convergence speed test.
        relax_cycle = lambda x: hm.relax_cycle.relax_cycle(multilevel, 1.0, 1, 1, 100).run(x)
        # FMG start so (xbda) has a reasonable initial guess.
        x = hm.bootstrap_eigen.fmg(multilevel, num_cycles_finest=0)
        x, conv_factor = hm.run.run_iterative_eigen_method(level.operator, relax_cycle, x, 20, print_frequency=1)

        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.316, 1e-2)

    def test_2_level_two_bootstrap_steps_same_speed_as_one(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap_eigen.generate_test_matrix(a, 0, num_examples=4, num_bootstrap_steps=2)
        assert len(multilevel) == 2

        level = multilevel.finest_level

        # Convergence speed test.
        relax_cycle = lambda x: hm.eigensolver.relax_cycle(multilevel, 1.0, 4, 3, 100).run(x)
        # FMG start so x has a reasonable initial guess.
        logger.info("2-level convergence test")
        x = hm.bootstrap_eigen.fmg(multilevel, num_cycles_finest=0)

        # Add some random noise but still stay near a reasonable initial guess.
        # x += 0.1 * np.random.random(x.shape)
        # multilevel.finest_multilevel.lam *= 1.01

        x, conv_factor = hm.run.run_iterative_eigen_method(level.operator, relax_cycle, x, 20,
                                                          print_frequency=1, residual_stop_value=1e-11)

        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.304, 1e-2)

    def test_2_level_bootstrap_least_squares_interpolation_laplace(self):
        n = 16
        kh = 0
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap_eigen.generate_test_matrix(a, 0, num_examples=4, interpolation_method="ls",
                                                                     num_sweeps=1000)
        assert len(multilevel) == 2

        level = multilevel.finest_level
        # Convergence speed test.
        relax_cycle = lambda x: hm.eigensolver.relax_cycle(multilevel, 1.0, 1, 1, 100).run(x)
        # FMG start so x has a reasonable initial guess.
        x = hm.bootstrap_eigen.fmg(multilevel, num_cycles_finest=0)
        x, conv_factor = hm.run.run_iterative_eigen_method(level.operator, relax_cycle, x, 20, print_frequency=1)

        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.316, 1e-2)

    def test_2_level_bootstrap_least_squares_interpolation(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap_eigen.generate_test_matrix(a, 0, num_examples=4, interpolation_method="ls")
        assert len(multilevel) == 2

        level = multilevel.finest_level
        # Convergence speed test.
        relax_cycle = lambda x: hm.eigensolver.relax_cycle(multilevel, 1.0, 1, 1, 100).run(x)
        # FMG start so x has a reasonable initial guess.
        x = hm.bootstrap_eigen.fmg(multilevel, num_cycles_finest=0)
        x, conv_factor = hm.run.run_iterative_eigen_method(level.operator, relax_cycle, x, 20, print_frequency=1)

        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.316, 1e-2)

    #@unittest.skip("3-level not working well, solve 2-level well enough first.")
    def test_3_level_fixed_domain(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap_eigen.generate_test_matrix(
            a, 0, num_sweeps=20, num_examples=4, initial_max_levels=3)
        assert len(multilevel) == 3

        level = multilevel.finest_level

        # Convergence speed test.
        # FMG start so x has a reasonable initial guess.
        x_init = hm.bootstrap_eigen.fmg(multilevel, num_cycles_finest=0, num_cycles=1)
        #        multilevel.lam = exact_eigenpair(level.a)

        relax_cycle = lambda x: hm.eigensolver.relax_cycle(multilevel, 1.0, 1, 1, 100, num_levels=3).run(x)
        x, conv_factor = hm.run.run_iterative_eigen_method(level.operator, relax_cycle, x_init, 15)
        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.32, 1e-2)
