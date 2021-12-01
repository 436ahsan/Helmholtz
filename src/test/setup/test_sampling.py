import logging
import sys
import numpy as np

import helmholtz as hm
import helmholtz.repetitive.coarsening_repetitive as cr

logger = logging.getLogger("nb")


class TestCoarseningUniform:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.set_printoptions(precision=6, linewidth=1000)
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(stream=sys.stdout, level=logging.WARN, format="%(levelname)-8s %(message)s",
                            datefmt="%a, %d %b %Y %H:%M:%S")
        np.random.seed(1)

    def test_get_windows_by_index(self):
        n = 10
        num_functions = 4
        aggregate_size = 4
        stride = 2
        num_windows = 21
        x = np.arange(num_functions * n).reshape(num_functions, n).transpose()

        x_windows = hm.setup.sampling.get_windows_by_index(x, np.arange(aggregate_size), stride, num_windows)

        assert x_windows.shape == (num_windows, aggregate_size)
        assert np.array_equal(x_windows, np.array([
             [0, 1, 2, 3],
             [10, 11, 12, 13],
             [20, 21, 22, 23],
             [30, 31, 32, 33],
             [2, 3, 4, 5],
             [12, 13, 14, 15],
             [22, 23, 24, 25],
             [32, 33, 34, 35],
             [4, 5, 6, 7],
             [14, 15, 16, 17],
             [24, 25, 26, 27],
             [34, 35, 36, 37],
             [6, 7, 8, 9],
             [16, 17, 18, 19],
             [26, 27, 28, 29],
             [36, 37, 38, 39],
             [8, 9, 0, 1],
             [18, 19, 10, 11],
             [28, 29, 20, 21],
             [38, 39, 30, 31],
             [0, 1, 2, 3]
        ]))

        # r = cr.Coarsener(np.array([[1, 0]]))
        # r = r.tile(n // 2)
        #
        # xc = r.dot(x)
        # xc_windows = hm.setup.sampling.get_windows(xc, aggregate_size, stride, num_windows)
        #
        # assert xc_windows.shape == (num_windows, aggregate_size)
