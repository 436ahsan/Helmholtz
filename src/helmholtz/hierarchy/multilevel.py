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

    def __init__(self, a, b, relaxer, r, p, r_csr, p_csr):
        """
        Creates a level in the multilevel hierarchy.
        Args:
            a: level operator.
            b: level mass matrix.
            relaxer: relaxation execution object.
            r: coarsening operator (type of coarse variables that this level is).
            p: this-level-to-next-finer-level interpolation.
            r_csr: coarsening operator in CSR matrix form (for repetitive problems).
            p_csr: interpolation operator in CSR matrix form (for repetitive problems).
        """
        self.a = a
        self.b = b
        self.r = r
        self.p = p
        self._r_csr = r_csr
        self._p_csr = p_csr
        self._restriction_csr = p_csr.transpose() if p_csr is not None else None
        self._relaxer = relaxer

    @staticmethod
    def create_finest_level(a, relaxer):
        return Level(a, scipy.sparse.eye(a.shape[0]), relaxer, None, None, None, None)

    @property
    def size(self):
        """Returns the number of variables in this level."""
        return self.a.shape[0]

    def print(self):
        _LOGGER.info("a = \n" + str(self.a.toarray()))

        if isinstance(self._r_csr, scipy.sparse.csr_matrix):
            _LOGGER.info("r = \n" + str(self._r_csr.todense()))
        if isinstance(self._p_csr, scipy.sparse.csr_matrix):
            _LOGGER.info("p = \n" + str(self._p_csr.todense()))

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

    def operator(self, x: np.array, lam: float = 0) -> np.array:
        """
        Returns the operator action (A-lam*B)*x.
        Args:
            x: vector of size n or a matrix of size n x m, where A, B are n x n.

        Returns:
            (A-B*lam)*x.
        """
        if lam == 0:
            return self.a.dot(x)
        else:
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

    def relax(self, x: np.array, b: np.array, lam: float = 0.0) -> np.array:
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
        Returns the restriction action P^T*x.
        Args:
            x: vector of size n or a matrix of size n x m.

        Returns:
            P^T*x.
        """
        return self._restriction_csr.dot(x)

    def coarsen(self, x: np.array) -> np.array:
        """
        Returns the coarsening action R*x.
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
