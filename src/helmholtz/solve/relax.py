"""Kaczmarz relaxation."""
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


class KaczmarzRelaxer:
    """Implements Kaczmarz relaxation for (A-lam*B)*x = b."""

    def __init__(self, a: scipy.sparse.spmatrix, b: scipy.sparse.spmatrix = None, block_size: int = 1,
                 reverse: bool = False) -> None:
        """
        Creates a Kaczmarz relaxer for (A-lam*B)*x=b .
        Args:
            a: left-hand-side matrix.
            b: mass matrix, if non-None.
            reverse: iff True, run in reverse lex order instead of lex order.
        """
        if b is None:
            b = scipy.sparse.eye(a.shape[0])
        self._a = a.tocsr()
        self._at = a.T
        self._b = b.tocsr()
        self._bt = b.T
        # Storing M = lower triangular parts of (A-lam*B)*(A-lam*B)^T in CSR format (the Kaczmarz splitting matrix) for
        # linear solve efficiency.

        if reverse:
            if block_size == 1:
                splitter = scipy.sparse.triu
            else:
                # TODO(oren): To be implemented.
                splitter = lambda a: block_triu(a, block_size)
        else:
            if block_size == 1:
                splitter = scipy.sparse.tril
            else:
                splitter = lambda a: block_tril(a, block_size)

        self._ma = splitter(a.dot(self._at)).tocsr()
        self._m_cross = splitter(a.dot(self._bt) + b.dot(self._at)).tocsr()
        self._mb = splitter(b.dot(self._bt)).tocsr()
        self._at = self._at.tocsr()

    def step(self, x: np.array, b: np.array, lam: float = 0) -> np.array:
        """
            Executes a Kaczmarz sweep on A*x = b or (A-lam*B)*x=b (if self._b is non-None). The B-term is not frozen.
        Args:
            x: initial guess. May be a vector of size n or a matrix of size n x m, where A is n x n.
            b: RHS. Same size as x.
            lam: eigenvalue, if this is an eigenvalue problem.

        Returns:
            x after relaxation.
        """
        if lam == 0:
            # Optimization for lam=0.
            r = b - self._a.dot(x)
            # TODO(orenlivne): determine if it is better to do 3 tril solves and add them up instead of adding up the
            # matrices and doing one as done here.
            delta = scipy.sparse.linalg.spsolve(self._ma, r)
        else:
            r = b - self._a.dot(x) + lam * self._b.dot(x)
            # TODO(orenlivne): determine if it is better to do 3 tril solves and add them up instead of adding up the
            # matrices and doing one as done here.
            delta = scipy.sparse.linalg.spsolve(self._ma - lam * self._m_cross + lam ** 2 * self._mb, r)
        if delta.ndim < x.ndim:
            delta = delta[:, None]
        return x + self._at.dot(delta) - lam * self._bt.dot(delta)


class GsRelaxer:
    """Implements Gauss-Seidel relaxation for A*x=b."""

    def __init__(self, a: scipy.sparse.spmatrix, block_size: int = 1, omega: float = 1.0) -> None:
        """
        Creates a Gauss-Seidel relaxer for A*x=b.
        Args:
            a: left-hand-side matrix.
            block_size: block size, for block Gauss-Seidel.
        """
        self._a = a.tocsr()
        if block_size == 1:
            self._m = scipy.sparse.tril(a).tocsr()
        else:
            self._m = block_tril(a, block_size)
        self._omega = omega

    def step(self, x: np.array, b: np.array, lam: float = 0) -> np.array:
        """
            Executes a Gauss-Seidel sweep on A*x = b.
        Args:
            x: initial guess. May be a vector of size n or a matrix of size n x m, where A is n x n.
            b: RHS. Same size as x.

        Returns:
            x after relaxation.
        """
        delta = scipy.sparse.linalg.spsolve(self._m, b - self._a.dot(x))
        if delta.ndim < x.ndim:
            delta = delta[:, None]
        return x + self._omega * delta


def block_tril(a, block_size):
    block_sizes = np.array([block_size] * (a.shape[0] // block_size))
    end_index = np.concatenate(tuple(i * [ind] for i, ind in zip(block_sizes, np.cumsum(block_sizes))))
    lower_part = a.nonzero()[1] < end_index[a.nonzero()[0]]
    return scipy.sparse.csr_matrix((a.data[lower_part], (a.nonzero()[0][lower_part], a.nonzero()[1][lower_part])), shape=a.shape)


class SymmetricKaczmarzRelaxer:
    """Implements forward-backward Kaczmarz relaxation for (A-lam*B)*x = b."""

    def __init__(self, a: scipy.sparse.spmatrix, b: scipy.sparse.spmatrix, block_size: int = 1, num_sweeps: int = 1) -> None:
        self._relax_forward = KaczmarzRelaxer(a, b, block_size=block_size, reverse=False)
        self._relax_backward = KaczmarzRelaxer(a, b, block_size=block_size, reverse=True)
        self._num_sweeps = num_sweeps

    def step(self, x: np.array, b: np.array, lam: float = 0) -> np.array:
        for i in range(self._num_sweeps):
            x = self._relax_forward.step(x, b, lam=lam)
        for i in range(self._num_sweeps):
            x = self._relax_backward.step(x, b, lam=lam)
        return x
