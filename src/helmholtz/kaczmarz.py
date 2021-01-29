"""Kaczmarz relaxation."""
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


class KaczmarzRelaxer:
    """Implements Kaczmarz relaxation for A*x = 0."""
    def __init__(self, a: scipy.sparse.dia_matrix) -> None:
        """
        Creates a Kaczmarz relaxer.
        Args:
            a: left-hand-side matrix.
        """
        self._a = a
        self._at = a.transpose()
        # Storing M = lower triangular part of A*A^T in CSR format (the Kaczmarz splitting matrix) for linear solve
        # efficiency.
        self._m = scipy.sparse.tril(a.dot(self._at)).tocsr()

    def step(self, x: np.array, b: np.array) -> np.array:
        """
            Executes a Kaczmarz sweep on A*x = 0.
        Args:
            x: initial guess. May be a vector of size n or a matrix of size n x m, where A is n x n.
            b: RHS. Same size as x.

        Returns:
            x after relaxation.
        """
        delta = scipy.sparse.linalg.spsolve(self._m, b - self._a.dot(x))
        if delta.ndim < x.ndim:
            delta = delta[:, None]
        return x + self._at.dot(delta)
