import logging
import sys

import numpy as np
import pytest
import sklearn.metrics.pairwise

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

        level = multilevel.level[0]
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
        x = hm.multilevel.random_test_matrix((n,), num_examples=1)
        b = np.zeros_like(x)
#        method = lambda x: level.relax(x, b)
        multilevel = hm.multilevel.Multilevel()
        multilevel.level.append(level)
        # Run enough Kaczmarz relaxations per lambda update (not just 1 relaxation) so we converge to the minimal one.
        nu = 5
        method = lambda x: multilevel.relax_cycle(x, None, None, nu)
        x, conv_factor = hm.multilevel.relax_test_matrix(level.operator, level.rq, method, x, 100)

        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.09770, 1e-3)
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.0977884, 1e-3)
        # nu relaxations + lambda update convergence factor.
        assert conv_factor == pytest.approx(0.958, 1e-3)

    def test_2_level_one_bootstrap_step_improves_convergence(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a, 0, num_sweeps=10, num_examples=8, num_bootstrap_steps=1)
        assert len(multilevel) == 2

        level = multilevel.level[0]

        # Convergence speed test.
        relax_cycle = lambda x: multilevel.relax_cycle(x, 1, 1, 100, debug=False, update_lam="finest")
        # FMG start so (x, lambda) has a reasonable initial guess.
        x = hm.multilevel.random_test_matrix((n // 2,), num_examples=1)
        level.global_params.lam = 0
        logger.info("Level 1")
        logger.info("lam {}".format(level.global_params.lam))
        for _ in range(100):
            x = multilevel.relax_cycle(x, None, None, 10, finest_level_ind=1)
        x = multilevel.level[1].interpolate(x)
        logger.info("Level 0")
        logger.info("lam {}".format(level.global_params.lam))
        x, conv_factor = hm.multilevel.relax_test_matrix(level.operator, level.rq, relax_cycle, x, 20, print_frequency=1)

        assert level.global_params.lam == pytest.approx(0.0977590650225, 1e-3)
        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.45055035, 1e-2)

    def test_2_level_two_bootstrap_steps_same_speed_as_one(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a, 0, num_sweeps=10, num_examples=8, num_bootstrap_steps=2)
        assert len(multilevel) == 2

        level = multilevel.level[0]

        # Convergence speed test.
        relax_cycle = lambda x: multilevel.relax_cycle(x, 1, 1, 100, debug=False, update_lam="finest")
        # FMG start so (x, lambda) has a reasonable initial guess.
        x = hm.multilevel.random_test_matrix((n // 2,), num_examples=1)
        level.global_params.lam = 0
        logger.debug("Level 1 lam {}".format(level.global_params.lam))
        for _ in range(100):
            x = multilevel.relax_cycle(x, None, None, 10, finest_level_ind=1)
        x = multilevel.level[1].interpolate(x)
        logger.debug("Level 0 lam {}".format(level.global_params.lam))
        x, conv_factor = hm.multilevel.relax_test_matrix(level.operator, level.rq, relax_cycle, x, 20, print_frequency=1)

        assert level.global_params.lam == pytest.approx(0.0977590650225, 1e-3)
        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        assert conv_factor == pytest.approx(0.46037, 1e-2)
