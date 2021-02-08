"""Basic solver (residual, relaxation) of the 2D Helmholtz operator with Homogeneous RHS, periodic B.C. and
constant k."""
import numpy as np


class Helmholtz2d:
    """Basic solver (residual, relaxation) of the 1D Helmholtz operator with Homogeneous RHS, periodic B.C. and
    constant k."""

    def __init__(self, k: float, h: float):
        """
        Constructs a 1D Helmholtz operator with homogeneous RHS and periodic B.C.. Note that we do not specify the
        number of gridpoints; calls to operator() and relax_*() receive inputs whose shape determines that, so they
        can be reused for different domain shapes with the same meshsize.

        Args:
            k: wave number.
            h: mesh size.
        """
        self.k2 = k ** 2
        self.h = h

    def operator(self, u: np.ndarray) -> np.ndarray:
        """
        Returns the residual of u: -L(u), where L = Delta + k^2 I.

        Args:
            u: n x m x k test matrix, where n x m = #gridpoints in domain and k=#test functions.

        Returns:
            a: n x m x k residual matrix, where n x m = #gridpoints in domain and k=#test functions.
        """
        m, n = u.shape[:2]
        a = np.zeros(u.shape)
        diagonal = 4 - self.k2 * self.h ** 2
        for i in range(m):
            for j in range(n):
                a[i, j] = -u[(i - 1) % m, j] - u[(i + 1) % m, j] - u[i, (j - 1) % n] \
                          - u[i, (j + 1) % n] + diagonal * u[i, j]
        return a

    def relax_kaczmarz(self, u: np.ndarray) -> None:
        """
        Executes Kaczmarz relaxation in-place on each column of u. This is guaranteed to reduce the L2 norm of u.

        Args:
            u: n x m x k test matrix, where n x m = #gridpoints in domain and k=#test functions.

        Returns:
            None. u is updated in place.
        """
        m, n = u.shape[:2]
        diagonal = 4 - self.k2 * self.h ** 2
        for i in range(m):
            for j in range(n):
                r = -(-u[(i - 1) % m, j] - u[(i + 1) % m, j] - u[i, (j - 1) % n] \
                      - u[i, (j + 1) % n] + diagonal * u[i, j])
                delta = r / (diagonal ** 2 + 4)
                u[i, j] += diagonal * delta
                u[(i - 1) % m, j] -= delta
                u[(i + 1) % m, j] -= delta
                u[i, (j - 1) % n] -= delta
                u[i, (j + 1) % n] -= delta

    def relax_gs(self, u: np.ndarray) -> None:
        """
        Executes Gauss-Seidel relaxation in-place on each column of u. (This is guaranteed to reduce the energy of u
        for k = 0.)

        Args:
            u: n x m x k test matrix, where n x m = #gridpoints in domain and k=#test functions.

        Returns:
            None. u is updated in place.
        """
        m, n = u.shape[:2]
        diagonal = 4 - self.k2 * self.h ** 2
        for i in range(m):
            for j in range(n):
                u[i, j] = (u[(i - 1) % m, j] + u[(i + 1) % m, j] +
                           u[i, (j - 1) % n] + u[i, (j + 1) % n]) / diagonal
