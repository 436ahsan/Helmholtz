"""Restriction (R) construction routines. Based on SVD on an aggregate."""
import numpy as np
import scipy.sparse


class Restrictor:
    """
    Encapsulates the restruction operator as both a full array over an aggregate (for easy tiling) and sparse CSR matrix
    format. Assumes non-overlapping aggregate (block sparsity) structure."""

    def __init__(self, r: np.ndarray):
        """
        Creates an interpolation into a window (aggregate).
        Args:
            r: {aggregate_size} x {aggregate_size} coarse variable definition over an aggregate. Includes all possible
                coarse vars, out of which we select nc based on an energy threshold in tile().
        """
        self._r = r

    def asarray(self) -> scipy.sparse.csr_matrix:
        """ Returns the dense restriction matrix on an aggregate."""
        return self._r

    def tile(self, n: int) -> scipy.sparse.csr_matrix:
        """
        Returns a tiled restriction over an n-times larger periodic domain, as a CSR matrix.
        Args:
            n: number of times to tile the interpolation = #aggregates in the domain.

        Returns: the sparse CSR interpolation matrix.
        """
        return hm.linalg.tile_array(self.asarray(), n)


def create_restriction(x_aggregate_t, threshold: float) -> scipy.sparse.csr_matrix:
    """
    Generates R (coarse variables) on an aggregate from  SVD principcal components.

    Args:
        x_aggregate_t: fine-level test matrix on an aggregate, transposed.
        threshold: relative reconstruction error threshold. Determines nc.

    Returns:
        restriction operator nc x {aggregate_size}, list of all singular values on aggregate.
    """
    _, s, vh = svd(x_aggregate_t)
    nc = _get_interpolation_caliber(np.array([threshold]))[0]
    return Restrictor(r[:nc]), s


def _relative_reconstruction_error(s):
    """Returns the fit squared error for a singular value array 's'."""
    fit_squared_error = np.concatenate((np.cumsum(s[::-1] ** 2)[::-1], [0]))
    if fit_squared_error[0] == 0.0:
        fit_squared_error[0] = 1.0
    return (fit_squared_error / fit_squared_error[0]) ** 0.5


def _get_interpolation_caliber(s: np.ndarray, threshold: float) -> np.ndarray:
    """
    Returns the number of principal components (IW) corresponding to a relative reconstruction error.
    Args:
        s: array-like, [N,] singular values of a matrix.
        threshold: array-like, [K] list of relative reconstruction error thresholds (values of t).

    Returns: IW: array, [K], Number of principal components corresponding to each element of 'threshold'.
    """
    return np.array([np.where( _relative_reconstruction_error(s) <= t)[0][0] for t in threshold])
