import logging
import sys
import numpy as np
from numpy.ma.testutils import assert_array_almost_equal

import helmholtz as hm
import helmholtz.repetitive.coarsening_repetitive as cr

logger = logging.getLogger("nb")


class TestCoarseningRepetitive:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.set_printoptions(precision=6, linewidth=1000)
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(stream=sys.stdout, level=logging.WARN, format="%(levelname)-8s %(message)s",
                            datefmt="%a, %d %b %Y %H:%M:%S")
        np.random.seed(1)

    def test_repetitive_coarsening(self):
        n = 32
        kh = 0.6
        num_sweeps = 100
        aggregate_size = 4
        a = hm.linalg.helmholtz_1d_operator(kh, n)

        # Generate relaxed test matrix.
        level = hm.setup.hierarchy.create_finest_level(a)
        x = hm.solve.run.random_test_matrix((n,))
        b = np.zeros_like(x)
        x, _ = hm.solve.run.run_iterative_method(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)

        # Generate coarse variables (R) based on a window of x.
        x_aggregate_t = x[:aggregate_size].T
        r, _ = cr.create_coarsening(x_aggregate_t, 0.1)

        # Convert to sparse matrix + tile over domain.
        assert r.asarray().shape == (2, 4)
        r_csr = r.tile(n // aggregate_size)

        assert r_csr.shape == (16, 32)

    def test_repetitive_coarsening_is_same_in_different_windows(self):
        n = 32
        kh = 0.1 # 0.6
        num_sweeps = 100
        aggregate_size = 4
        a = hm.linalg.helmholtz_1d_operator(kh, n)

        # Generate relaxed test matrix.
        level = hm.setup.hierarchy.create_finest_level(a)
        x = hm.solve.run.random_test_matrix((n,))
        b = np.zeros_like(x)
        x, _ = hm.solve.run.run_iterative_method(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)

        # Generate coarse variables (R) based on different windows of x.
        # Note: all coarsenings and singular values will be almost identical except the two windows (offset = 29, 30)
        # due to Kaczmarz stopping at point 31 (thus 30, 31, 1 are co-linear).
        r_by_offset = np.array([hm.linalg.normalize_signs(
            cr.create_coarsening(
                hm.linalg.get_windows_by_index(x, np.arange(offset, aggregate_size + offset), 1, 1).T, 0.1)[0].asarray())
            for offset in range(len(x))])
        # R should not change much across different windows.
        mean_entry_error = np.mean(((np.std(r_by_offset, axis=0) / np.mean(np.abs(r_by_offset), axis=0)).flatten()))
        assert mean_entry_error <= 0.03

    def test_create_coarsening_full_domain(self):
        n = 16
        kh = 0.6
        num_sweeps = 100
        aggregate_size = 4
        a = hm.linalg.helmholtz_1d_operator(kh, n)

        # Generate relaxed test matrix.
        level = hm.setup.hierarchy.create_finest_level(a)
        x = hm.solve.run.random_test_matrix((n,))
        b = np.zeros_like(x)
        x, _ = hm.solve.run.run_iterative_method(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)

        # Generate coarse variables (R) on the non-repetitive domain.
        r, aggregates, nc, energy_error = cr.create_coarsening_domain(x, threshold=0.15)

        assert [a.tolist() for a in aggregates] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        assert r.shape == (8, 16)
        # To print r:
        # print(','.join(np.array2string(y, separator=",", formatter={'float_kind':lambda x: "%.2f" % x})
        # for y in np.array(r.todense())))
        assert_array_almost_equal(
            r.todense(), [
            [0.58, 0.67, 0.46, 0.08, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [-0.47, -0.01, 0.47, 0.75, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, -0.46, -0.62, -0.56, -0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.60, 0.20, -0.33, -0.71, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.44, 0.63, 0.57, 0.29, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.60, -0.19, 0.33, 0.70, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, -0.37, -0.58, -0.59, -0.43],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.65, 0.30, -0.23, -0.65]
        ], decimal=2)
