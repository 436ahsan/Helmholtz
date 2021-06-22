"""coarsening (R) construction routines. Based on SVD on an aggregate."""
import logging
import numpy as np
import scipy.sparse
from numpy.linalg import svd
from typing import List, Tuple

import helmholtz as hm

_LOGGER = logging.getLogger(__name__)


class Coarsener:
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
        """ Returns the dense coarsening matrix on an aggregate."""
        return self._r

    def tile(self, n: int) -> scipy.sparse.csr_matrix:
        """
        Returns a tiled coarsening over an n-times larger periodic domain, as a CSR matrix.
        Args:
            n: number of times to tile the interpolation = #aggregates in the domain.

        Returns: the sparse CSR interpolation matrix.
        """
        return hm.linalg.tile_array(self.asarray(), n)


def create_coarsening(x_aggregate_t, threshold: float) -> Tuple[Coarsener, np.ndarray]:
    """
    Generates R (coarse variables) on an aggregate from SVD principcal components.

    Args:
        x_aggregate_t: fine-level test matrix on an aggregate, transposed.
        threshold: relative reconstruction error threshold. Determines nc.

    Returns:
        coarsening operator nc x {aggregate_size}, list of all singular values on aggregate.
    """
    u, s, vh = svd(x_aggregate_t)
    nc = _get_interpolation_caliber(s, np.array([threshold]))[0]
    return Coarsener(vh[:nc]), s


def create_coarsening_full_domain(x, threshold: float = 0.1, max_coarsening_ratio: float = 0.5,
                                  max_aggregate_size: int = 8) -> Tuple[scipy.sparse.csr_matrix, List[np.ndarray]]:
    """
    Creates the next coarse level's SVD coarsening operator R on a full domain (non-repetitive).
    Args:
        x: fine-level test matrix.
        threshold: relative reconstruction error threshold. Determines nc.
        max_coarsening_ratio: maximum allowed coarsening ratio. If exceeded at a certain aggregate size, we double
            it until it is reached (or when the aggregate size becomes too large, in which case an exception is raised).
        max_aggregate_size: maximum allowed aggregate size. If exceeded, an exception is thrown.

    Returns: coarsening operator R, list of aggregates.
    """
    # Sweep the domain left to right; add an aggregate and its coarsening until we get to the domain end.
    start = 0
    r_aggregate = []
    aggregates = []
    while start < x.shape[0]:
        r = _create_aggregate_coarsening(x, threshold, max_coarsening_ratio, max_aggregate_size, start)
        r_aggregate.append(r)
        aggregate_size = r.shape[1]
        aggregates.append(np.arange(start, start + aggregate_size))
        start += aggregate_size

    # Merge all aggregate coarsening operators.
    return scipy.sparse.block_diag(r_aggregate).tocsr(), aggregates


def _create_aggregate_coarsening(x, threshold, max_coarsening_ratio, max_aggregate_size, start):
    """
    Creates the next coarse level's SVD coarsening operator R.
    Args:
        x: fine-level test matrix.
        threshold: relative reconstruction error threshold. Determines nc.
        max_coarsening_ratio: maximum allowed coarsening ratio. If exceeded at a certain aggregate size, we double
            it until it is reached (or when the aggregate size becomes too large, in which case an exception is raised).
        max_aggregate_size: maximum allowed aggregate size. If exceeded, an exception is thrown.
        start: start index of aggregate.

    Returns: R of the aggregate.
    """
    domain_size = x.shape[0]
    # Increase aggregate size until we reach a small enough coarsening ratio.
    aggregate_expansion_factor = 2
    aggregate_size, coarsening_ratio = 1, 1
    # NOTE: domain is assumed to contain at least two points.
    end = start + aggregate_size
    coarsening_by_aggregate_size = {aggregate_size: np.ones((1, 1))}
    # While the aggregate has room for expansion nor reached the end of the domain, and we haven't obtained the target
    # coarsening ratio yet, expand the aggregate and calculate a 'threshold'-tolerance SVD coarsening.
    while (aggregate_size <= max_aggregate_size // aggregate_expansion_factor) and (end < x.shape[0]) and \
            (coarsening_ratio > max_coarsening_ratio):
        aggregate_size *= aggregate_expansion_factor
        end = min(start + aggregate_size, domain_size)
        x_aggregate_t = x[start:end].transpose()
        r, s = hm.setup.coarsening.create_coarsening(x_aggregate_t, threshold)
        r = r.asarray()
        # n = actual aggregate size after trimming to domain end. nc = #coarse variables.
        nc, n = r.shape
        coarsening_by_aggregate_size[n] = r
        coarsening_ratio = nc / n
        _LOGGER.debug("SVD {:2d} x {:2d} nc {} cr {:.2f} err {:.3f} sigma {}"
                      " err {}".format(x_aggregate_t.shape[0], x_aggregate_t.shape[1], nc, coarsening_ratio,
                                         (sum(s[nc:] ** 2) / sum(s ** 2)) ** 0.5,
                                         np.array2string(s, separator=", ", precision=2),
                                         np.array2string(
                                             (1 - np.cumsum(s ** 2) / sum(s ** 2)) ** 0.5, separator=", ",
                                             precision=2)))
    r, n = min(((r, n) for n, r in coarsening_by_aggregate_size.items()), key=lambda item: item[0].shape[0] / item[1])
    if r.shape[0] / aggregate_size > max_coarsening_ratio:
        _LOGGER.warning("Could not find a good coarsening ratio for aggregate {}:{}, n {} nc {} cr {:.2f}".format(
            start, start + n, n, r.shape[0], r.shape[0] / n
        ))
    return r


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
    return np.array([np.where(_relative_reconstruction_error(s) <= t)[0][0] for t in threshold])
