import numpy as np
import pytest

import helmholtz.analysis


class TestIdeal:
    def test_find_singular_kh(self):
        np.random.seed(0)
        n = 96
        root = helmholtz.analysis.ideal.find_singular_kh("5-point", n)

        assert root[0] == pytest.approx(0.52339, 1e-5)
        assert np.abs(root[1]) < 1e-15
