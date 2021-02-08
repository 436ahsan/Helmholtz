"""Relaxation for the homogenous Laplacian."""
import numpy as np


class Solver:
    def __init__(self, k):
        self.k = k

    def laplacian(self, u):
        m, n = u.shape[:2]
        a = np.zeros(u.shape)
        self.set_homogeneous_bc(a)
        e = self.e
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                a[i, j] = -2 * (1 + e) * u[i, j] \
                          + u[i - 1, j] + u[i + 1, j] \
                          + e * (u[i, j - 1] + u[i, j + 1])
        return a

    def set_homogeneous_bc(self, u):
        u[0] = 0
        u[-1] = 0
        u[:, 0, :] = 0
        u[:, -1, :] = 0

    def relax_kaczmarz(self, u):
        m, n = u.shape
        d_inv = 1 / (4 * (1 + e))
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                u[i, j] = d_inv * (u[(i - 1) % m, j] + u[(i + 1) % m, j] + u[i, (j - 1) % n] + u[i, (j + 1) % n])
