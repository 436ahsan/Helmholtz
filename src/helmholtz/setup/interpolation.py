"""Interpolation construction routines. Fits interpolation to 1D Helmholtz test functions in particular (with specific
coarse neighborhoods based on domain periodicity)."""
import numpy as np
import scipy.sparse
from typing import Tuple

import helmholtz as hm
import helmholtz.setup.interpolation_fit


class Interpolator:
    """
    Encapsulates the interpolation as both relative-location neighbor list (for easy tiling) and sparse CSR matrix
    format. In contrast to the coarsening operator, which is assumed to be non-overlapping and thus a simple CSR
    matrix, this object gives us greater flexibility in interpolating from neighboring aggregates."""

    def __init__(self, nbhr: np.ndarray, data: np.ndarray, nc: int):
        """
        Creates an interpolation into a window (aggregate).
        Args:
            nbhr: coarse-level variable interpolatory set. nbhr[i] is the set of the fine variable i.
            data: the corresponding interpolation coefficients.
            nc: number of coarse variables per aggregate. In 1D at the finest level, we know there are two principal
                components, so nc=2 makes sense there.
        """
        self._nbhr = nbhr
        self._data = data
        self._nc = nc

    def asarray(self):
        return self._nbhr, self._data

    def __len__(self):
        """Returns the number of fine variables being interpolated to (making up a single aggregate)."""
        return len(self._nbhr)

    def tile(self, n: int) -> scipy.sparse.csr_matrix:
        """
        Returns a tiled interpolation over an n-times larger periodic domain, as a CSR matrix.
        Args:
            n: number of times to tile the interpolation = #aggregates in the domain.

        Returns: the sparse CSR interpolation matrix.
        """
        # Build P sparsity arrays for a single aggregate.
        nc = self._nc
        aggregate_size = len(self)
        row = np.tile(np.arange(aggregate_size)[:, None], self._nbhr[0].shape).flatten()
        col = np.concatenate([nbhr_i for nbhr_i in self._nbhr])

        # Tile P of a single aggregate over the entire domain.
        tiled_row = np.concatenate([row + aggregate_size * ic for ic in range(n)])
        # Periodically wrap around coarse variable indices.
        tiled_col = np.concatenate([col + nc * ic for ic in range(n)]) % (nc * n)
        tiled_data = np.tile(self._data.flatten(), n)
        domain_size = aggregate_size * n
        return scipy.sparse.coo_matrix((tiled_data, (tiled_row, tiled_col)), shape=(domain_size, nc * n)).tocsr()


def create_interpolation(method: str, r: np.ndarray, x_aggregate_t: np.ndarray, xc_t: np.ndarray, domain_size: int,
                         nc: int, caliber: int) -> Interpolator:
    """
    Creates an interpolation operator.
    Args:
        method: type of interpolation: "svd" (R^T) or "ls" (regularized least-squares fitting).
        r:
        x_aggregate_t: fine-level test matrix over the aggregate, transposed.
        xc_t: coarse-level test matrix over entire domain, transposed.
        domain_size: number of fine gridpoints in domain,
        nc: number of coarse variables per aggregate. In 1D at the finest level, we know there are two principal
            components, so nc=2 makes sense there.
        caliber: interpolation caliber.

    Returns:
        interpolation object.
    """
    if method == "svd":
        return Interpolator(np.tile(np.arange(nc, dtype=int)[:, None], r.shape[1]).transpose(),
                            r.transpose(), nc)
    elif method == "ls":
        return create_interpolation_least_squares(x_aggregate_t, xc_t, domain_size, nc, caliber)
    else:
        raise Exception("Unsupported interpolation method '{}'".format(method))


def create_interpolation_least_squares(x_aggregate_t: np.ndarray, xc_t: np.ndarray, domain_size: int, nc: int,
                                       caliber: int) -> Interpolator:
    """Defines interpolation to an aggregate by LS fitting to coarse neighbors of each fine var. The global
    interpolation P is the tiling of the aggregate P over the domain."""

    # Define nearest coarse neighbors of each fine variable.
    aggregate_size = x_aggregate_t.shape[1]
    num_aggregates = domain_size // aggregate_size
    num_coarse_vars = nc * num_aggregates
    # Find nearest neighbors of each fine point in an aggregate.
    nbhr = np.mod(_geometric_neighbors(aggregate_size, nc), num_coarse_vars)
    nbhr = _sort_neighbors_by_similarity(x_aggregate_t, xc_t, nbhr)

    # Fit interpolation over an aggregate.
    alpha = np.array([0, 0.001, 0.01, 0.1, 1.0])
    num_examples = x_aggregate_t.shape[0]
    fitter = helmholtz.setup.interpolation_fit.InterpolationFitter(
        x_aggregate_t, xc=xc_t, nbhr=nbhr,
        fit_samples=num_examples // 3, val_samples=num_examples // 3, test_samples=num_examples // 3)
    error, alpha_opt = fitter.optimized_relative_error(caliber, alpha, return_weights=True)
    # Interpolation validation error = error[:, 1]
    data = error[:, 2:]
    return hm.setup.interpolation.Interpolator(nbhr[:, :caliber], data, nc)


def create_interpolation_least_squares_auto_nbhrs(
        x: np.ndarray, a: scipy.sparse.csr_matrix, r: scipy.sparse.csr_matrix) -> \
        Tuple[scipy.sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates the interpolation operator P by least squares fitting from r*x to x. Interpolatory sets are automatically
    determined by a's graph.
    Args:
        x_t: fine-level test matrix. shape = (num_fine_vars, num_examples).
        a: fine-level operator (only its adjacency graph is used herer).
        r: coarsening operator.

    Returns:
        interpolation matrix P,
        relative fit error at all fine points,
        relative validation error at all fine points,
        relative test error at all fine points,
        optimal alpha for all fine points.
    """
    # Define interpolation neighbors. We use as much neighbors of each fine point in an aggregate such that the coarse
    # stencil is not increased beyond its size if we only interpolated the from aggregate's coarse variables. This is
    # the union of all coarse variables in all aggregates of the points in i's fine-level (a) stencil.
    nbhr = [np.unique(r[:, a[i].nonzero()[1]].nonzero()[0]) for i in range(x.shape[0])]
    # Ridge regularization parameter (list of values).
    alpha = np.array([0, 0.01, 0.1, 0.1, 1])
    num_examples = x.shape[1]
    return hm.setup.interpolation_fit.create_interpolation_least_squares(
        x.transpose(), r.dot(x).transpose(), nbhr, alpha,
        fit_samples=int(0.5 * num_examples),
        val_samples=int(0.25 * num_examples),
        test_samples=int(0.25 * num_examples))


def _geometric_neighbors(w: int, nc: int):
    """
    Returns the relative indices of the interpolatory set of each fine variable in a window.
    Args:
        w: size of window (aggregate).
        nc: number of coarse variables per aggregate. In 1D at the finest level, we know there are two principal
            components, so nc=2 makes sense there.

    Returns: array of size w x {num_neighbors} of coarse neighbor indices (relative to fine variable indices) of each
        fine variable.
    """
    # Here we assume w points per aggregate and the same number mc of coarse vars per aggregate, but in general
    # aggregate sizes may vary.
    fine_var = np.arange(w, dtype=int)

    # Index of neighboring coarse variable. Left neighboring aggregate for points on the left half of the window;
    # right for right.
    coarse_nbhr = np.zeros_like(fine_var)
    left = fine_var < w // 2
    right = fine_var >= w // 2
    coarse_nbhr[left] = -1
    coarse_nbhr[right] = 1

    # All nbhrs = central aggregate coarse vars + neighboring aggregate coarse vars.
    coarse_vars_center = np.tile(np.arange(nc, dtype=int), (w, 1))
    return np.concatenate((coarse_vars_center, coarse_vars_center + nc * coarse_nbhr[:, None]), axis=1)


def _sort_neighbors_by_similarity(x_aggregate: np.array, xc: np.array, nbhr: np.array):
    return np.array([nbhr_of_i[
                         np.argsort(-_similarity(x_aggregate[:, i][:, None], xc[:, nbhr_of_i]))
                     ][0] for i, nbhr_of_i in enumerate(nbhr)])


def _similarity(x, xc):
    """Returns all pairwise cosine similarities between the two test matrices. Does not zero out the mean, which
    assumes intercept = False in fitting interpolation.
    """
    return hm.linalg.pairwise_cos_similarity(x, xc, squared=True)
