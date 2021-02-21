import logging
import sys

import numpy as np

import helmholtz as hm

logger = logging.getLogger("nb")


class TestRestriction:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.set_printoptions(precision=6, linewidth=1000)
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(levelname)-8s %(message)s",
                            datefmt="%a, %d %b %Y %H:%M:%S")
        np.random.seed(1)

    def test_restriction(self):
        n = 32
        kh = 0.6
        num_sweeps = 100
        aggregate_size = 4
        a = hm.linalg.helmholtz_1d_operator(kh, n)

        # Generate relaxed test matrix.
        level = hm.multilevel.Level.create_finest_level(a)
        x = hm.run.random_test_matrix((n,))
        b = np.zeros_like(x)
        lam = 0
        x, _, _ = hm.run.relax_test_matrix(level.operator, lambda x, lam: (level.relax(x, b, lam), lam),
                                           x, lam, num_sweeps=num_sweeps)

        # Generate coarse variables (R) based on a window of x.
        x_aggregate_t = x[:aggregate_size].transpose()
        r, _ = hm.restriction.create_restriction(x_aggregate_t, 0.1)

        # Convert to sparse matrix + tile over domain.
        assert r.asarray().shape == (2, 4)
        r_csr = r.tile(n // aggregate_size)

        assert r_csr.shape == (16, 32)

    def test_restriction_is_same_in_different_windows(self):
        n = 32
        kh = 0.1 # 0.6
        num_sweeps = 100
        aggregate_size = 4
        a = hm.linalg.helmholtz_1d_operator(kh, n)

        # Generate relaxed test matrix.
        level = hm.multilevel.Level.create_finest_level(a)
        x = hm.run.random_test_matrix((n,))
        b = np.zeros_like(x)
        lam = 0
        x, _, _ = hm.run.relax_test_matrix(level.operator, lambda x, lam: (level.relax(x, b, lam), lam),
                                           x, lam, num_sweeps=num_sweeps)

        # Generate coarse variables (R) based on different windows of x.
        # Note: all restrictions and singular values will be almost identical except the two windows (offset = 29, 30)
        # due to Kaczmarz stopping at point 31 (thus 30, 31, 1 are co-linear).
        r_by_offset = np.array([hm.linalg.normalize_signs(
            hm.restriction.create_restriction(
                hm.linalg.get_window(x, offset, aggregate_size).transpose(), 0.1)[0].asarray())
            for offset in range(len(x))])
        # R should not change much across different windows.
        mean_entry_error = np.mean(np.abs((np.std(r_by_offset, axis=0) / np.mean(np.abs(r_by_offset), axis=0)).flatten()))
        assert mean_entry_error <= 0.03
