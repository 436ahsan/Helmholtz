import logging
import sys

import numpy as np
import pytest
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

    def test_bootstrap_1level(self):
        n = 16
        kh = 0.5

        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a, 0, num_sweeps=100)

        assert x.shape == (16, 64)

        assert len(multilevel) == 2

        level = multilevel.finest_level
        assert level.a.shape == (16, 16)
        level.print()

        coarse_level = multilevel.level[1]
        assert coarse_level.a.shape == (8, 8)
        assert coarse_level._r_csr.shape == (8, 16)
        assert coarse_level._p_csr.shape == (16, 8)
        coarse_level.print()

    def test_run_1_level_relax(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = hm.multilevel.Level.create_finest_level(a)
        multilevel = hm.multilevel.Multilevel(level)
        x = hm.run.random_test_matrix((n,), num_examples=1)
        b = np.zeros_like(x)
        #        method = lambda x: level.relax(x, b)
        multilevel = hm.multilevel.Multilevel(level)
        # Run enough Kaczmarz relaxations per lambda update (not just 1 relaxation) so we converge to the minimal one.
        nu = 5
        lam = 0
        method = lambda x: hm.eigensolver.eigen_cycle(multilevel, 1.0, None, None, nu).run((x, lam))
        x, conv_factor = hm.run.relax_test_matrix(level.operator, level.rq, method, x, 100)

        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.09770, 1e-3)
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.0977884, 1e-3)
        # (nu relaxations + lambda update) convergence factor, so not very impressive.
        assert conv_factor == pytest.approx(0.807, 1e-3)

    def test_2_level_one_bootstrap_step_improves_convergence(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a, 0, num_examples=8)
        assert len(multilevel) == 2

        level = multilevel.finest_level
        lam = multilevel.lam
        # Convergence speed test.
        eigen_cycle = lambda x: hm.eigensolver.eigen_cycle(multilevel, 1.0, 1, 1, 100).run((x, lam))
        # FMG start so (x, lambda) has a reasonable initial guess.
        x = hm.bootstrap.fmg(multilevel, num_cycles_finest=0)
        x, conv_factor = hm.run.relax_test_matrix(level.operator, level.rq, eigen_cycle, x, 20, print_frequency=1)

        assert multilevel.lam == pytest.approx(0.0977590650225, 1e-3)
        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.269, 1e-2)

    def test_2_level_two_bootstrap_steps_same_speed_as_one(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a, 0, num_examples=8, num_bootstrap_steps=2)
        assert len(multilevel) == 2

        level = multilevel.finest_level

        # Convergence speed test.
        lam = multilevel.lam
        eigen_cycle = lambda x: hm.eigensolver.eigen_cycle(multilevel, 1.0, 4, 3, 100).run((x, lam))
        # FMG start so (x, lambda) has a reasonable initial guess.
        logger.info("2-level convergence test")
        x = hm.bootstrap.fmg(multilevel, num_cycles_finest=0)

        # Add some random noise but still stay near a reasonable initial guess.
        # x += 0.1 * np.random.random(x.shape)
        # multilevel.finest_multilevel.lam *= 1.01

        x, conv_factor = hm.run.relax_test_matrix(level.operator, level.rq, eigen_cycle, x, 20,
                                                  print_frequency=1, residual_stop_value=1e-11)

        assert multilevel.lam == pytest.approx(0.0977590650225, 1e-3)
        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.304, 1e-2)

    def test_3_level_fixed_domain(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a, 0, num_sweeps=10, num_examples=20, initial_max_levels=3)
        assert len(multilevel) == 3

        level = multilevel.finest_level

        # Convergence speed test.
        # FMG start so (x, lambda) has a reasonable initial guess.
        x_init = hm.bootstrap.fmg(multilevel, num_cycles_finest=0, num_cycles=1)
        #        multilevel.lam = exact_eigenpair(level.a)

        lam = multilevel.lam
        eigen_cycle = lambda x: hm.eigensolver.eigen_cycle(multilevel, 1.0, 1, 1, 100, num_levels=3).run((x, lam))
        x, conv_factor = hm.run.relax_test_matrix(level.operator, level.rq, eigen_cycle, x_init, 15)
        assert multilevel.lam == pytest.approx(0.0977590650225, 1e-3)
        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.32, 1e-2)


def exact_eigenpair(a):
    """Returns the exact minimum norm eigenvalue of the matrix a, for debugging."""
    lam, v = eig(a.toarray())
    lam = np.real(lam)
    ind = np.argsort(lam)
    lam = lam[ind]
    return v[:, ind[-2]], lam[-2]
