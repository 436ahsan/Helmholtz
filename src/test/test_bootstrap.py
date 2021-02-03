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
        x = hm.multilevel.random_test_matrix((n,))
        b = np.zeros_like(x)
        x, conv_factor = hm.multilevel.relax_test_matrix(level.operator, level.rq, lambda x: level.relax(x, b), x, 100)
        print(hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x))
        #assert hm.linalg.scaled_norm_of_matrix(x).mean() == pytest.approx(0.028129, 1e-3)

        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.09770, 1e-3)
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.116295694, 1e-3)
        assert conv_factor == pytest.approx(0.995, 1e-3)

    def test_2_level_relax_cycle_faster_than_1_level(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a, 0, num_sweeps=10, num_examples=8)

        relax_cycle = lambda x: multilevel.relax_cycle(x, 2, 2, 4)
        level = multilevel.level[0]
        x, conv_factor = hm.multilevel.relax_test_matrix(level.operator, level.rq, relax_cycle, x, 100)
        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.09770, 1e-3)
        assert (hm.linalg.scaled_norm_of_matrix(a.dot(x)) / hm.linalg.scaled_norm_of_matrix(x)).mean() == \
               pytest.approx(0.098258, 1e-3)
        # TODO(orenlivne): measure convergence factor differrently - how fast we converge to the eigenvalue seems to
        # be noisy.
        assert conv_factor == pytest.approx(0.879, 1e-3)

    def test_2_level_bootstrap_improves_2_level_convergence(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a, 0, num_sweeps=10, num_examples=8, num_bootstrap_steps=2)
        assert len(multilevel) == 2

        relax_cycle = lambda x: multilevel.relax_cycle(x, 1, 1, 100, debug=False, update_lam=False)
        level = multilevel.level[0]
#        x = hm.multilevel.random_test_matrix((n,))
        x = x[:, 0][:, None]

        import scipy.linalg
        level.global_params.lam = np.sort(np.real(scipy.linalg.eig(a.toarray())[0]))[-2]
        print("exact lam", level.global_params.lam)

        x, conv_factor = hm.multilevel.relax_test_matrix(level.operator, level.rq, relax_cycle, x, 20, print_frequency=1)

        assert level.global_params.lam == pytest.approx(0.0977590650225, 1e-3)
        assert np.mean([level.rq(x[:, i]) for i in range(x.shape[1])]) == pytest.approx(0.097759, 1e-3)
        # TODO(orenlivne): measure convergence factor differrently - how fast we converge to the eigenvalue seems to
        # be noisy.
        assert conv_factor == pytest.approx(0.735, 1e-3)
