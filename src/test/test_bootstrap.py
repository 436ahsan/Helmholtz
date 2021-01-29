import logging
import numpy as np
import helmholtz as hm
import sklearn.metrics.pairwise
import sys
from numpy.ma.testutils import assert_array_almost_equal
from numpy.linalg import norm

logger = logging.getLogger("nb")


class TestBootstrap:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.set_printoptions(precision=6, linewidth=1000)
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(levelname)-8s %(message)s",
                            datefmt="%a, %d %b %Y %H:%M:%S")
        np.random.seed(0)

    def test_generate_test_matrix(self):
        n = 16
        kh = 0.5

        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a, num_sweeps=100)

        assert x.shape == (16, 48)

        assert len(multilevel) == 2

        level = multilevel.level[0]
        assert level.a.shape == (16, 16)
        level.print()

        coarse_level = multilevel.level[1]
        assert coarse_level.a.shape == (8, 8)
        assert coarse_level.r.shape == (8, 16)
        assert coarse_level.p.shape == (16, 8)
        coarse_level.print()

    def test_run_1_level_relax(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = hm.multilevel.Level(a)
        x = hm.multilevel.random_test_matrix((n, ))
        b = np.zeros_like(x)
        x = hm.multilevel.relax_test_matrix(level.operator, lambda x: level.relax(x, b), x, 10)

    def test_run_2_level_relax_cycle(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a)

        relax_cycle = lambda x: multilevel.relax(x, 2, 2, 4)
        x = hm.multilevel.relax_test_matrix(multilevel.level[0].operator, relax_cycle, x, 100)

    def test_distances_between_window_and_coarse_vars(self):
        n = 32
        kh = 0.6
        nc = 1
        num_sweeps = 100
        aggregate_size = 4
        a = hm.linalg.helmholtz_1d_operator(kh, n)

        # Generate relaxed test matrix.
        level = hm.multilevel.Level(a)
        x = hm.multilevel.random_test_matrix((n, ))
        b = np.zeros_like(x)
        x = hm.multilevel.relax_test_matrix(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)

        # Generate coarse variables (R) based on a window of x.
        x_aggregate_t = x[:aggregate_size].transpose()
        r, _ = hm.bootstrap.create_coarse_vars(x_aggregate_t, n, nc)
        xc = r.dot(x)
        xc_t = xc.transpose()

        # Measure distance between x of an aggregate and xc.
        d = sklearn.metrics.pairwise.cosine_similarity(x_aggregate_t.transpose(), xc_t.transpose())

        assert d.shape == (4, 8)
