import logging
import sys

import numpy as np
import pytest
import sklearn.metrics.pairwise
from numpy.ma.testutils import assert_array_almost_equal

import helmholtz as hm

logger = logging.getLogger("nb")


class TestRestriction:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.set_printoptions(precision=6, linewidth=1000)
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(levelname)-8s %(message)s",
                            datefmt="%a, %d %b %Y %H:%M:%S")
        np.random.seed(0)

    def test_resrtriction(self):
        n = 32
        kh = 0.6
        nc = 1
        num_sweeps = 100
        aggregate_size = 4
        a = hm.linalg.helmholtz_1d_operator(kh, n)

        # Generate relaxed test matrix.
        level = hm.multilevel.Level.create_finest_level(a)
        x = hm.multilevel.random_test_matrix((n,))
        b = np.zeros_like(x)
        x, _ = hm.multilevel.relax_test_matrix(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)

        # Generate coarse variables (R) based on a window of x.
        x_aggregate_t = x[:aggregate_size].transpose()
        r, _ = hm.restriction.create_restriction(x_aggregate_t, 0.1)

        # Convert to sparse matrix + tile over domain.
        assert r.asarray().shape == (2, 4)
        r_csr = r.tile(n // aggregate_size)

        assert r_csr.shape == (16, 32)
