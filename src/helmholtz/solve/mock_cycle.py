"""Mock cycle for predicting the quality of a coarse-variable set."""
import helmholtz as hm
import numpy as np
import scipy.sparse


class MockCycle:
    r"""Implements mock cycle comprising of relaxation sweeps + idealized coarse-level correction."""
    def __init__(self,
                 relaxer,
                 q: scipy.sparse.spmatrix,
                 num_steps: int = 1,
                 splitting: str = "direct",
                 num_corrector_steps: int = 1,
                 omega: float = 1.0):
        """
        Creates a mock cycle solve method.

        Args:
            relaxer: relaxation method.
            q: coarsening (a.k.a. aggregation) matrix Q.
            splitting: type of splitting of Q*Q^T where Q = coarsening matrix. Vallues: 'direct' or 'jacobi'.
            num_steps (int): number of GD steps per mock cycle.
            num_corrector_steps(int): number of coarse level corrector steps per mock cycle.
            omega: corrector relaxation damping parameter, if not direct.
        """
        self._num_steps = num_steps
        self._num_corrector_steps = num_corrector_steps
        self._relaxer = relaxer
        # For linear problems, we set the coarse variables to 0 (doesn't matter what they are).
        baseline_coarse_values = np.zeros((q.shape[0], 1))
        self._corrector = MockCorrector(q, baseline_coarse_values, splitting=splitting, omega=omega)

    def __call__(self, x):
        """Performs a mock cycle for A*x=0 on the test vector or matrix x."""
        b = np.zeros_like(x)
        # Perform 'num_steps' optimizer steps.
        for i in range(self._num_steps):
            x = self._relaxer(x, b)
        for i in range(self._num_corrector_steps):
            x = self._corrector(x)
        return x


class MockCorrector:
    r"""Implements the mock coarse-level-correction step of the Mock Cycle (MC) for a single parameter, for either
    input or output activation coarsening."""
    def __init__(self, q, baseline_coarse_values, splitting: str = "direct", omega: float = 1.0):
        """
        Creates a mock coarse-level correction step applicator.

        Args:
            q: scipy CSR sparse aggregation matrix Q: nc x n.
            baseline_coarse_values: (CPU numpy array) of flattened baseline coarse values to keep the fine values
                compatible with.
            splitting: type of matrix splitting to apply in the compatibility step ('direct'|'jacobi').
            omega: relaxation damping parameter.
        """
        self._splitting = _create_splitting(splitting, q)
        self._omega = omega
        qt = np.transpose(q)
        self._q = q
        self._qt = self._q.T
        self._q0 = baseline_coarse_values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Performs a mock coarse-level correction step that approximately restores the coarse variables to the value
        'self._baseline_coarse_values'.

        This is an exact solver step that minimizes the correction's norm such that the coarse variable values are
        restored after it is applied, that is,

        x <- x - Q^T*(Q*Q^T)^{-1}*(Q*x - q0)

        where q0 = baseline coarse values.

        Returns: updated x.

        """
        r = self._q0 - self._q.dot(x)
        delta = self._omega * self._qt.dot(self._splitting.solve(r))
        return x + delta


def _create_splitting(splitting: str, q: scipy.sparse.spmatrix):
    """
    Returns a solver of the correction step equation M*y = r, as part of the Kaczmarz compatibility step.
    Here Q*Q^T is split into M and Q*Q^T - M.
    Args:
        splitting: kind of splitting ('direct', 'jacobi').
        q: the matrix Q (the aggregation matrix in this case)

    Returns: a _Splitting instance.
    """
    if splitting == "direct":
        # Perform a sparse LU decomposition of Q*Q^T on the CPU and cache it for fast repeated linear solves.
        # Note: Q*Q^T = I for SVD construction, so we don't need to invert in this case.
        return hm.linalg.SparseLuSolver(q.dot(np.transpose(q)))
    elif splitting == "jacobi":
        return _JacobiSplitting(q)
    # TODO(orenlivne): add "gs" - triangular solve. May be faster than Jacobi/w-Jacobi.
    else:
        raise Exception("Unsupported splitting kind '%s'" % (splitting,))


class _Splitting:
    r"""Solves the correction step equation M*y = r, as part of the Kaczmarz compatibility step. Here Q*Q^T is split
    into M and Q*Q^T - M."""

    def solve(self, r: np.ndarray) -> np.ndarray:
        """
        Solves M*y = r, where r = Q*w-q0 is the current residual of the correction step

        w <- w - Q^T*M^{-1}*(Q*w - q0)

        Args:
            r: nc x num_sheaves residual tensor.

        Returns: M^{-1}*r.
        """
        raise Exception("Must be implemented by sub-classes.")


class _JacobiSplitting(_Splitting):
    r"""Solves the correction step equation diag(Q*Q^T)*y = r (Jacobi splitting)."""
    def __init__(self, q: scipy.sparse.spmatrix):
        """
        Creates a Jacobi matrix splitter.

        Args:
            q: scipy CSR sparse aggregation matrix Q: nc x n.
        """
        self._d_inv = 1 / np.ndarray((q.dot(q.T)).diagonal())

    def solve(self, r: np.ndarray) -> np.ndarray:
        """
        Solves diag(Q*Q^T)*y = r, where r = Q*w-q0 is the current residual of the correction step.

        Args:
            r: nc x num_sheaves residual tensor.

        Returns:  diag(Q*Q^T)^{-1}*r.
        """
        return self._d_inv[:, None] * r
