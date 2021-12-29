import numpy as np
import scipy.sparse
from numpy.linalg import norm
from numpy.ma.testutils import assert_array_almost_equal

import helmholtz as hm


class TestRelax:

    def test_kaczmarz_sweep_reduces_residual_l2_norm(self):
        """Tests the matrix implementation of Kaczmarz against pointwise and the invariant that the L2 norm of the
        residual decreases (up to the null-space components' amplitude)."""
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        relaxer = hm.solve.relax.KaczmarzRelaxer(a, scipy.sparse.eye(a.shape[0]))

        x = np.random.random((n, 5))
        b = np.zeros_like(x)
        r_norm = norm(a.dot(x))
        for i in range(10):
            y = kaczmarz_relax_with_loop(kh, x)
            x = relaxer.step(x, b)
            assert_array_almost_equal(x - x.mean(), y - y.mean())
            r_norm_new = norm(a.dot(x))
            assert r_norm_new < r_norm
            r_norm = r_norm_new

    def test_gauss_seidel_sweep_same_as_loop_implementation(self):
        n = 16
        kh = 0.5
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        relaxer = hm.solve.relax.GsRelaxer(a)

        x = np.random.random((n, 5))
        b = np.zeros_like(x)
        for i in range(10):
            y = gs_relax_with_loop(kh, x)
            x = relaxer.step(x, b)
            assert_array_almost_equal(x - x.mean(), y - y.mean())

    def test_gauss_seidel_sweep_reduces_energy_for_spd(self):
        n = 16
        kh = 0
        # Make a positive definite (= -Laplacian).
        a = -hm.linalg.helmholtz_1d_operator(kh, n)
        relaxer = hm.solve.relax.GsRelaxer(a)

        x = np.random.random((n, 1))
        b = np.zeros_like(x)
        energy = x.T.dot(a.dot(x))[0, 0]
        for i in range(10):
            y = gs_relax_with_loop(kh, x)
            x = relaxer.step(x, b)
            assert_array_almost_equal(x - x.mean(), y - y.mean())
            energy_new = x.T.dot(a.dot(x))[0, 0]
            assert energy_new < energy
            energy = energy_new


def kaczmarz_relax_with_loop(kh, x: np.ndarray) -> np.ndarray:
    """
    Executes Kaczmarz relaxation using a loop for the 1D Helmholtz operator A, which does not require A's storage.

    Args:
        x: n x k test matrix, where n = #gridpoints and k=#test functions.

    Returns:
        None. u is updated in place.
    """
    u = x.copy()
    n = u.shape[0]
    diagonal = 2 - kh ** 2
    for i in range(n):
        r = -(-u[(i - 1) % n] - u[(i + 1) % n] + diagonal * u[i])
        delta = r / (diagonal ** 2 + 2)
        u[i] += diagonal * delta
        u[(i - 1) % n] -= delta
        u[(i + 1) % n] -= delta
    return u


def gs_relax_with_loop(kh, x: np.ndarray) -> np.ndarray:
    """
    Executes Gauss-Seidel relaxation using a loop for the 1D Helmholtz operator A, which does not require A's storage.

    Args:
        x: n x k test matrix, where n = #gridpoints and k=#test functions.

    Returns:
        None. u is updated in place.
    """
    u = x.copy()
    n = u.shape[0]
    diagonal = 2 - kh ** 2
    for i in range(n):
        r = -(-u[(i - 1) % n] - u[(i + 1) % n] + diagonal * u[i])
        u[i] += r / diagonal
    return u
