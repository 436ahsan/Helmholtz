import logging
import numpy as np
import helmholtz as hm
import sys
from numpy.ma.testutils import assert_array_almost_equal
from numpy.linalg import norm


class TestBootstrap:

    def test_generate_functions_and_fit_interpolation(self):
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="%(levelname)-8s %(message)s",
                            datefmt="%a, %d %b %Y %H:%M:%S")
        logger = logging.getLogger("nb")

        n = 16
        kh = 0.5

        a = hm.linalg.helmholtz_1d_operator(kh, n)

        e = hm.bootstrap.generate_test_functions(a, (4, ))

        print(e.shape)