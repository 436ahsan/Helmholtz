"""Multilevel solver (producer of low-residual test functions of the Helmholtz operator."""
import logging
from typing import Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.linalg import norm

import helmholtz as hm
from helmholtz.linalg import scaled_norm

_LOGGER = logging.getLogger("multilevel")


class Level:
    """A single level in the multilevel hierarchy."""

    def __init__(self, a, b, r, p, r_csr, p_csr):
        self.a = a
        self.b = b
        self.r = r
        self.p = p
        self._r_csr = r_csr
        self._p_csr = p_csr
        self._relaxer = hm.relax.KaczmarzRelaxer(a, b)

    @staticmethod
    def create_finest_level(a):
        return Level(a, scipy.sparse.eye(a.shape[0]), None, None, None, None)

    @staticmethod
    def create_coarse_level(a, b, r, p):
        num_aggregates = a.shape[0] // r.asarray().shape[1]
        r_csr = r.tile(num_aggregates)
        p_csr = p.tile(num_aggregates)
        # Form Galerkin coarse-level operator.
        ac = (r_csr.dot(a)).dot(p_csr)
        bc = (r_csr.dot(b)).dot(p_csr)
        return Level(ac, bc, r, p, r_csr, p_csr)

    def print(self):
        _LOGGER.info("a = \n" + str(self.a.toarray()))
        if self.r is not None:
            _LOGGER.info("r = \n" + str(self.r))
        if self.p is not None:
            _LOGGER.info("p = \n" + str(self.p))

    def stiffness_operator(self, x: np.array) -> np.array:
        """
        Returns the operator action A*x.
        Args:
            x: vector of size n or a matrix of size n x m, where A is n x n.

        Returns:
            A*x.
        """
        return self.a.dot(x)

    def mass_operator(self, x: np.array) -> np.array:
        """
        Returns the operator action B*x.
        Args:
            x: vector of size n or a matrix of size n x m, where B is n x n.

        Returns:
            B*x.
        """
        return self.b.dot(x)

    def operator(self, x: np.array, lam) -> np.array:
        """
        Returns the operator action (A-lam*B)*x.
        Args:
            x: vector of size n or a matrix of size n x m, where A, B are n x n.

        Returns:
            (A-B*lam)*x.
        """
        return self.a.dot(x) - lam * self.b.dot(x)

    def normalization(self, x: np.array) -> np.array:
        """
        Returns the eigen-normalization functional (Bx, x).
        Args:
            x: vector of size n or a matrix of size n x m, where A, B are n x n.

        Returns:
           (Bx, x) for each column of x.
        """
        return np.array([(self.b.dot(x[:, i])).dot(x[:, i]) for i in range(x.shape[1])])

    def rq(self, x: np.array, b: np.array = None) -> np.array:
        """
        Returns the Rayleigh Quotient of x.
        Args:
            x: vector of size n or a matrix of size n x m, where A, B are n x n.
            b: RHS vector, for FAS coarse problems.

        Returns:
           (Ax, x) / (Bx, x) or ((Ax - b), x) / (Bx, x) if b is not None.
        """
        if b is None:
            return (self.a.dot(x)).dot(x) / (self.b.dot(x)).dot(x)
        else:
            return (self.a.dot(x) - b).dot(x) / (self.b.dot(x)).dot(x)

    def relax(self, x: np.array, b: np.array, lam) -> np.array:
        """
        Executes a relaxation sweep on A*x = 0 at this level.
        Args:
            x: initial guess. May be a vector of size n or a matrix of size n x m, where A is n x n.
            b: RHS. Same size as x.

        Returns:
            x after relaxation.
        """
        return self._relaxer.step(x, b, lam=lam)

    def restrict(self, x: np.array) -> np.array:
        """
        Returns the restriction action R*x.
        Args:
            x: vector of size n or a matrix of size n x m.

        Returns:
            x^c = R*x.
        """
        return self._r_csr.dot(x)

    def interpolate(self, xc: np.array) -> np.array:
        """
        Returns the interpolation action R*x.
        Args:
            xc: vector of size n or a matrix of size nc x m.

        Returns:
            x = P*x^c.
        """
        return self._p_csr.dot(xc)


class Multilevel:
    """The multilevel hierarchy. Contains a sequence of levels."""

    def __init__(self, finest_level: Level):
        """
        Creates an initial multi-level hierarchy with one level.
        Args:
            finest_level: finest Level.
        """
        self.level = [finest_level]

    def __len__(self):
        return len(self.level)

    @property
    def finest_level(self) -> Level:
        """
        Returns the finest level.

        Returns: finest level object.
        """
        return self.level[0]
