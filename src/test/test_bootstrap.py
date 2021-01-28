import logging
import numpy as np
import helmholtz as hm
import sys
from numpy.ma.testutils import assert_array_almost_equal
from numpy.linalg import norm

logger = logging.getLogger("nb")


class TestBootstrap:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.set_printoptions(precision=3, linewidth=1000)
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(levelname)-8s %(message)s",
                            datefmt="%a, %d %b %Y %H:%M:%S")
        np.random.seed(0)

    def test_generate_test_matrix(self):
        n = 16
        kh = 0.5

        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a)

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

    def test_run_2_level_relax_cycle(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        x, multilevel = hm.bootstrap.generate_test_matrix(a)

        relax_cycle = lambda x: multilevel.relax(x, 2, 2, 4)
        x = hm.multilevel.relax_test_matrix(multilevel.level[0].operator, relax_cycle, x, 100)

