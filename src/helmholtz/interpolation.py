"""Interpolation construction routines. Fits interpolation to 1D Helmholtz test functions in particular (with specific
coarse neighborhoods based on domain periodicity)."""
import numpy as np
import scipy.sparse
import sklearn.metrics.pairwise

import helmholtz as hm


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
        row = np.tile(np.arange(aggregate_size)[:, None], nc).flatten()
        col = np.concatenate([nbhr_i for nbhr_i in self._nbhr])

        # Tile P of a single aggregate over the entire domain.
        tiled_row = np.concatenate([row + aggregate_size * ic for ic in range(n)])
        tiled_col = np.concatenate([col + nc * ic for ic in range(n)])
        tiled_data = np.tile(self._data.flatten(), n)
        domain_size = aggregate_size * n
        return scipy.sparse.coo_matrix((tiled_data, (tiled_row, tiled_col)),
                                       shape=(domain_size, nc * n)).tocsr()


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
        return Interpolator(np.tile(np.arange(nc, dtype=int)[:, None], x_aggregate_t.shape[1]).transpose(),
                            r.transpose(), nc)
    elif method == "ls":
        return _create_interpolation_ls(x_aggregate_t, xc_t, domain_size, nc, caliber)
    else:
        raise Exception("Unsupported interpolation method '{}'".format(method))


def _create_interpolation_ls(x_aggregate_t: np.ndarray, xc_t: np.ndarray, domain_size: int, nc: int, caliber: int) -> \
        Interpolator:
    """Defines interpolation to an aggregate by LS fitting to coarse neighbors of each fine var. The global
    interpolation P is the tiling of the aggregate P over the domain."""

    # Define nearest coarse neighbors of each fine variable.
    aggregate_size = x_aggregate_t.shape[1]
    num_aggregates = domain_size // aggregate_size
    nbhr = hm.interpolation_build.geometric_neighbors(domain_size, aggregate_size, nc)
    nbhr = hm.interpolation_build.sort_neighbors_by_similarity(x_aggregate_t, xc_t, nbhr, num_aggregates)

    # Fit interpolation over an aggregate.
    alpha = np.array([0, 0.001, 0.01, 0.1, 1.0])
    num_examples = x_aggregate_t.shape[0]
    fitter = hm.interpolation_fit.InterpolationFitter(
        x_aggregate_t, xc=xc_t, nbhr=nbhr,
        fit_samples=num_examples // 3, val_samples=num_examples // 3, test_samples=num_examples // 3)
    error, alpha_opt = fitter.optimized_relative_error(caliber, alpha, return_weights=True)
    # Interpolation validation error = error[:, 1]
    data = np.concatenate([pi for pi in error[:, 2:]])
    return hm.interpolator.Interpolator(nbhr, data, nc)


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
    # Here there are aggregate_shape points per aggregate and the same number of coarse vars per aggregate, but in
    # general aggregate sizes may vary.
    fine_var = np.arange(0, w, dtype=int)
    # Index of aggregate that i belongs to, for each i in fine_var.
    aggregate_of_fine_var = fine_var // w
    num_coarse_vars = nc * num_aggregates
    # Global index of coarse variables with each aggregate.
    aggregate_coarse_vars = np.array([np.arange(ic, ic + nc, dtype=int) for ic in range(0, num_coarse_vars, nc)])
    nbhr = [None] * w
    for fine_var in range(w):
        center_aggregate = fine_var // w
        if fine_var < w // 2:
            # Left neighboring aggregate is nearest neighbor of the fine var.
            nbhr_aggregate = center_aggregate - 1
        else:
            # Right neighboring aggregate is nearest neighbor of the fine var.
            nbhr_aggregate = center_aggregate + 1
        # Use center aggregate and neighboring aggregate.
        # coarse_nbhrs = np.union1d(aggregate_coarse_vars[center_aggregate], aggregate_coarse_vars[nbhr_aggregate])
        # Use center aggregate only.
        coarse_nbhrs = aggregate_coarse_vars[center_aggregate]
        # print(fine_var, center_aggregate, nbhr_aggregate, coarse_nbhrs)
        nbhr[fine_var] = coarse_nbhrs
    return nbhr


def _sort_neighbors_by_similarity(x_aggregate: np.array, xc: np.array, nbhr: np.array, num_aggregates: int):
    return np.array([nbhr_of_i[
                         np.argsort(-similarity(x_aggregate[:, i][:, None], xc[:, nbhr_of_i % num_aggregates]))
                     ][0] for i, nbhr_of_i in enumerate(nbhr)])


def _similarity(x, xc):
    """Returns all pairwise cosine similarities between the two test matrices. Does not zero out the mean, which
    assumes intercept = False in fitting interpolation.
    """
    return sklearn.metrics.pairwise.cosine_similarity(x.transpose(), xc.transpose())
