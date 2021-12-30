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

    def __init__(self, a, b, relaxer, r, p, q):
        """
        Creates a level in the multilevel hierarchy.
        Args:
            a: level operator.
            b: level mass matrix.
            relaxer: relaxation execution object.
            r: coarsening operator (type of coarse variables that this level is).
            p: this-level-to-next-finer-level interpolation.
            q: restriction operator. Usually P^T, but could be different (e.g., r).
        """
        self.a = a
        self.b = b
        self._r = r
        self._p = p
        self._q = q
        self._relaxer = relaxer

    @staticmethod
    def create_finest_level(a, relaxer) -> "Level":
        return Level(a, scipy.sparse.eye(a.shape[0]), relaxer, None, None, None)

    @property
    def size(self):
        """Returns the number of variables in this level."""
        return self.a.shape[0]

    def print(self):
        _LOGGER.info("a = \n" + str(self.a.toarray()))

        if isinstance(self._r, scipy.sparse.csr_matrix):
            _LOGGER.info("r = \n" + str(self._r.todense()))
        if isinstance(self._p, scipy.sparse.csr_matrix):
            _LOGGER.info("p = \n" + str(self._p.todense()))

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
        return self._q.dot(x)

    def coarsen(self, x: np.array) -> np.array:
        """
        Returns the coarsening action R*x.
        Args:
            x: vector of size n or a matrix of size n x m.

        Returns:
            x^c = R*x.
        """
        return self._r.dot(x)

    def interpolate(self, xc: np.array) -> np.array:
        """
        Returns the interpolation action R*x.
        Args:
            xc: vector of size n or a matrix of size nc x m.

        Returns:
            x = P*x^c.
        """
        return self._p.dot(xc)


class Multilevel:
    """The multilevel hierarchy. Contains a sequence of levels."""

    def __init__(self):
        """
        Creates an empty multi-level hierarchy.
        """
        self._level = []

    @staticmethod
    def create(finest_level: Level) -> "Multilevel":
        """
        Creates an initial multi-level hierarchy with one level.

        Args:
            finest_level: finest Level.

        Returns: multilevel hierarchy with a single level.
        """
        multilevel = Multilevel()
        multilevel.add(finest_level)
        return multilevel

    def __len__(self) -> int:
        return len(self._level)

    def __iter__(self):
        return iter(self._level)

    def __getitem__(self, index: int) -> Level:
        return self._level[index]

    @property
    def finest_level(self) -> Level:
        """
        Returns the finest level.

        Returns: finest level object.
        """
        return self._level[0]

    def add(self, level: Level) -> None:
        """
        Adds a level to the multilevel hierarchy.
        Args:
            level: level to add.
        """
        self._level.append(level)

    def sub_hierarchy(self, finest: int) -> "Multilevel":
        """
        Returns the sub-hierarchy starting at level 'finest'.
        Args:
            finest: index of new finest level.

        Returns:
            Sub-hierarchy.
        """
        multilevel = Multilevel()
        for level in self._level[finest:len(self)]:
            multilevel.add(level)
        return multilevel
