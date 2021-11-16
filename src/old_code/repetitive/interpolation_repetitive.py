# TODO(oren): deprecate this module unless this is truly meant to be a repetitive framework with a local interolation
# construction. In that case, make sure to optimize the caliber as in the 'setup' package interpolation module.

"""Interpolation construction routines. Fits interpolation to 1D Helmholtz test functions in particular (with specific
coarse neighborhoods based on domain periodicity)."""
import logging
import numpy as np
import scipy.sparse
import helmholtz as hm

import helmholtz.setup.interpolation_ls_fit as interpolation_ls_fit
import helmholtz.setup.interpolation as interpolation


_LOGGER = logging.getLogger(__name__)


class Interpolator:
    """
    Encapsulates the interpolation as both relative-location neighbor list (for easy tiling) and sparse CSR matrix
    format. In contrast to the coarsening operator, which is assumed to be non-overlapping and thus a simple CSR
    matrix, this object gives us greater flexibility in interpolating from neighboring aggregates."""

    def __init__(self, nbhr: np.ndarray, data: np.ndarray, nc: int):
        """
        Creates an interpolation into a window (aggregate).
        Args:
            nbhr: coarse-level variable interpolation set. nbhr[i] is the set of the fine variable i.
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


def create_interpolation_repetitive(
        method: str, r: np.ndarray, x_aggregate_t: np.ndarray, xc_t: np.ndarray, domain_size: int,
        nc: int) -> Interpolator:
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

    Returns:
        interpolation object.
    """
    if method == "svd":
        return Interpolator(np.tile(np.arange(nc, dtype=int)[:, None], r.shape[1]).transpose(),
                            r.transpose(), nc)
    elif method == "ls":
        return _create_interpolation_least_squares_repetitive_auto(x_aggregate_t, xc_t, domain_size, nc)
    else:
        raise Exception("Unsupported interpolation method '{}'".format(method))


def _create_interpolation_least_squares_repetitive_auto(
        x_aggregate_t: np.ndarray, xc_t: np.ndarray, domain_size: int, nc: int,
        alpha: np.ndarray = np.array([0, 0.001, 0.01, 0.1, 1.0])) -> Interpolator:
    """Defines interpolation to an aggregate by LS fitting to coarse neighbors of each fine var. The global
    interpolation P is the tiling of the aggregate P over the domain."""

    # Define nearest coarse neighbors of each fine variable.
    num_examples, aggregate_size = x_aggregate_t.shape
    num_aggregates = domain_size // aggregate_size
    num_coarse_vars = nc * num_aggregates
    # Find nearest neighbors of each fine point in an aggregate.
    nbhr = np.mod(hm.setup.geometry.geometric_neighbors(aggregate_size, nc), num_coarse_vars)
    nbhr = interpolation.sort_neighbors_by_similarity(x_aggregate_t, xc_t, nbhr)

    return _create_interpolation_least_squares_repetitive(
        x_aggregate_t, xc_t, nbhr, nc, alpha=alpha,
        fit_samples=num_examples // 3, val_samples=num_examples // 3, test_samples=num_examples // 3)


def _create_interpolation_least_squares_repetitive(
        x_aggregate_t: np.ndarray, xc_t: np.ndarray, nbhr: np.ndarray, nc: int,
        alpha: np.ndarray = np.array([0, 0.001, 0.01, 0.1, 1.0]),
        fit_samples: int = 1000,
        val_samples: int = 1000,
        test_samples: int = 1000) -> Interpolator:
    """Defines interpolation to an aggregate by LS fitting to coarse neighbors of each fine var. The global
        interpolation P is the tiling of the aggregate P over the domain."""
    # Fit interpolation over an aggregate.
    fitter = interpolation_ls_fit.InterpolationLsFitter(
        x_aggregate_t, xc=xc_t, nbhr=nbhr, fit_samples=fit_samples, val_samples=val_samples, test_samples=test_samples)
    caliber = nbhr.shape[1]
    error, alpha_opt = fitter.optimized_relative_error(caliber, alpha, return_weights=True)
    # Interpolation validation error = error[:, 1]
    data = error[:, 2:]
    return Interpolator(nbhr[:, :caliber], data, nc)
