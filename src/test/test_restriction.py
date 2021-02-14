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
        np.random.seed(0)

    def test_restriction(self):
        n = 32
        kh = 0.6
        num_sweeps = 100
        aggregate_size = 4
        a = hm.linalg.helmholtz_1d_operator(kh, n)

        # Generate relaxed test matrix.
        level = hm.multilevel.Level.create_finest_level(a)
        x = hm.multilevel.random_test_matrix((n,))
        b = np.zeros_like(x)
        x, _ = hm.multilevel.relax_test_matrix(level.operator, level.rq,
                                               lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)

        # Generate coarse variables (R) based on a window of x.
        x_aggregate_t = x[:aggregate_size].transpose()
        r, _ = hm.restriction.create_restriction(x_aggregate_t, 0.1)

        # Convert to sparse matrix + tile over domain.
        assert r.asarray().shape == (2, 4)
        r_csr = r.tile(n // aggregate_size)

        assert r_csr.shape == (16, 32)

    def test_restriction_is_same_in_different_windows(self):
        n = 32
        kh = 0.6
        num_sweeps = 100
        aggregate_size = 4
        a = hm.linalg.helmholtz_1d_operator(kh, n)

        # Generate relaxed test matrix.
        level = hm.multilevel.Level.create_finest_level(a)
        x = hm.multilevel.random_test_matrix((n,))
        b = np.zeros_like(x)
        x, _ = hm.multilevel.relax_test_matrix(level.operator, level.rq,
                                               lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)

        # Generate coarse variables (R) based on different windows of x.
        r_by_offset = np.array([normalized_restriction(
            hm.restriction.create_restriction(get_window(x, aggregate_size, offset).transpose(), 0.1)[0])
            for offset in range(len(x))])
        # R should not change much across different windows.
        mean_entry_errror = np.mean(np.abs((np.std(r_by_offset, axis=0) / np.mean(r_by_offset, axis=0)).flatten()))
        assert mean_entry_errror <= 0.07


def get_window(x, aggregate_size, offset):
    # Wrap around window since we're on a periodic domain.
    x_aggregate = x[offset:min(offset + aggregate_size, len(x))]
    if offset + aggregate_size > len(x):
        x_aggregate = np.concatenate((x_aggregate, x[:offset + aggregate_size - len(x)]), axis=0)
    return x_aggregate


def normalized_restriction(r):
    r_array = r.asarray()
    return r_array * np.sign(r_array[:, 0])[:, None]