"""Interpolation construction routines. Fits interpolation to 1D Helmholtz test functions in particular (with specific
coarse neighborhoods based on domain periodicity)."""
import logging
import numpy as np
import scipy.sparse
from typing import Tuple
from scipy.linalg import norm

import helmholtz as hm


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
        caliber: interpolation caliber.

    Returns:
        interpolation object.
    """
    if method == "svd":
        return Interpolator(np.tile(np.arange(nc, dtype=int)[:, None], r.shape[1]).transpose(),
                            r.transpose(), nc)
    elif method == "ls":
        return create_interpolation_least_squares_repetitive(x_aggregate_t, xc_t, domain_size, nc)
    else:
        raise Exception("Unsupported interpolation method '{}'".format(method))


def create_interpolation_least_squares_repetitive(
        x_aggregate_t: np.ndarray, xc_t: np.ndarray, domain_size: int, nc: int,
        alpha: np.ndarray = np.array([0, 0.001, 0.01, 0.1, 1.0])) -> Interpolator:
    """Defines interpolation to an aggregate by LS fitting to coarse neighbors of each fine var. The global
    interpolation P is the tiling of the aggregate P over the domain."""

    # Define nearest coarse neighbors of each fine variable.
    num_examples, aggregate_size = x_aggregate_t.shape
    num_aggregates = domain_size // aggregate_size
    num_coarse_vars = nc * num_aggregates
    # Find nearest neighbors of each fine point in an aggregate.
    nbhr = np.mod(geometric_neighbors(aggregate_size, nc), num_coarse_vars)
    nbhr = _sort_neighbors_by_similarity(x_aggregate_t, xc_t, nbhr)

    return _create_interpolation_least_squares_repetitive(
        x_aggregate_t, xc_t, nbhr, nc, alpha=alpha,
        fit_samples=num_examples // 3, val_samples=num_examples // 3, test_samples=num_examples // 3)


def create_interpolation_least_squares_domain(
        x: np.ndarray, a: scipy.sparse.csr_matrix, r: scipy.sparse.csr_matrix,
        aggregate_size: int = None, nc: int = None, neighborhood: str = "extended", num_test_examples: int = 5,
        repetitive: bool = False,
        max_caliber: int = 6,
        target_interpolation_energy_error: float = 0.1) -> Tuple[scipy.sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates the interpolation operator P by least squares fitting from r*x to x. Interpolatory sets are automatically
    determined by a's graph.
    Args:
        x: fine-level test matrix. shape = (num_fine_vars, num_examples).
        a: fine-level operator (only its adjacency graph is used here).
        aggregate_size: size of aggregate (assumed uniform over domain).
        nc: number of coarse variables per aggregate.
        r: coarsening operator.
        neighborhood: "aggregate"|"extended" coarse neighborhood to interpolate from: only coarse variables in the
            aggregate, or the R*A*R^T sparsity pattern.
        num_test_examples: number of test functions dedicated to testing (do not participate in SVD, LS fit).
        repetitive: whether to exploit problem repetitiveness by creating a constant R stencil on all aggregates
            using windows from a single (or few) test vectors.
        max_caliber: maximum interpolation caliber to ty.
        target_interpolation_energy_error:

    Returns:
        interpolation matrix P.
        relative interpolation A-norm error on the test set.
        optimal caliber.
    """
    # Find nearest neighbors of each fine point in an aggregate.
    if repetitive:
        num_aggregates = int(np.ceil(a.shape[0] / aggregate_size))
        num_coarse_vars = nc * num_aggregates
        nbhr = np.mod(geometric_neighbors(aggregate_size, nc), num_coarse_vars)
    else:
        nbhr = get_neighbor_set(x, a, r, neighborhood)

    # Ridge regularization parameter (list of values).
    alpha = np.array([0, 0.01, 0.1, 0.1, 1])

    # Prepare fine and coarse test matrices.
    xc = r.dot(x)
    if repetitive:
        x_disjoint_aggregate_t, xc_disjoint_aggregate_t = \
            hm.setup.sampling.get_disjoint_windows(x, xc, aggregate_size, nc, max_caliber)
        nbhr = nbhr[:aggregate_size]
    else:
        x_disjoint_aggregate_t, xc_disjoint_aggregate_t = x.transpose(), xc.transpose()

    num_examples = x_disjoint_aggregate_t.shape[0]
    num_ls_examples = num_examples - num_test_examples
    val_samples = int(0.2 * num_ls_examples)
    fit_samples = num_examples - val_samples - num_test_examples
    folds = tuple(f.transpose() for f in
                  hm.linalg.create_folds(x.transpose(), (fit_samples, val_samples, num_test_examples)))
    # Fit interpolation for increasing caliber until the test error A-norm is below the accuracy threshold.
    for caliber in range(1, max_caliber + 1):
        # Create an interpolation over the samples: a single aggregate (if repetitive) or entire domain (otherwise).
        p = hm.setup.interpolation_fit.create_interpolation_least_squares(
            x_disjoint_aggregate_t, xc_disjoint_aggregate_t, np.array([n[:caliber] for n in nbhr]),
            alpha=alpha, fit_samples=fit_samples, val_samples=val_samples, test_samples=num_test_examples)
        if repetitive:
            p = _tile_interpolation_matrix(p, aggregate_size, nc, x.shape[0])

        error_l2 = np.array([np.mean(norm(f - p.dot(r.dot(f)), axis=0) / norm(f, axis=0)) for f in folds])
        error_a = np.array([np.mean(norm(a.dot(f - p.dot(r.dot(f))), axis=0) / norm(f, axis=0)) for f in folds])
        _LOGGER.debug("caliber {} error_l2 {} error_a {}".format(
            caliber,
            np.array2string(error_l2, separator=", ", precision=2),
            np.array2string(error_a, separator=", ", precision=2)))
        if error_a[-1] < target_interpolation_energy_error:
            return p

    raise Exception("Could not find a good caliber for threshold {}".format(target_interpolation_energy_error))


def get_neighbor_set(x, a, r, neighborhood):
    # Define interpolation neighbors. We use as much neighbors of each fine point in an aggregate such that the coarse
    # stencil is not increased beyond its size if we only interpolated the from aggregate's coarse variables. This is
    # the union of all coarse variables in all aggregates of the points in i's fine-level (a) stencil.
    if neighborhood == "aggregate":
        nbhr = [r[:, i].nonzero()[0] for i in range(x.shape[0])]
    elif neighborhood == "extended":
        nbhr = [np.unique(r[:, a[i].nonzero()[1]].nonzero()[0]) for i in range(x.shape[0])]
    else:
        raise Exception("Unsupported neighborhood type {}".format(neighborhood))
    return nbhr


def _tile_interpolation_matrix(p, aggregate_size, nc, domain_size):
    """Tiles 'p' over all aggregates. The last two aggregates may overlap."""
    num_aggregates = int(np.ceil(domain_size / aggregate_size))
    # TODO(oren): do not calculate in the entire xc_disjoint_aggregate_t and nbhr, which are currently global.
    # Just calculate the localization of these quantities for a single aggregate.
    num_coarse_vars = nc * num_aggregates
    row_agg, col_agg = p.nonzero()
    row_ind = np.concatenate(
        tuple(row_agg + start for start in hm.linalg.get_uniform_aggregate_starts(domain_size, aggregate_size)))
    col_ind = np.mod(
        np.concatenate(tuple(col_agg + nc * agg_index for agg_index in range(num_aggregates))),
        num_coarse_vars)
    data = np.tile(p.data, num_aggregates)
    return scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(domain_size, num_coarse_vars))


def _create_interpolation_least_squares_repetitive(
        x_aggregate_t: np.ndarray, xc_t: np.ndarray, nbhr: np.ndarray, nc: int,
        alpha: np.ndarray = np.array([0, 0.001, 0.01, 0.1, 1.0]),
        fit_samples: int = 1000,
        val_samples: int = 1000,
        test_samples: int = 1000) -> Interpolator:
    """Defines interpolation to an aggregate by LS fitting to coarse neighbors of each fine var. The global
        interpolation P is the tiling of the aggregate P over the domain."""
    # Fit interpolation over an aggregate.
    fitter = hm.setup.interpolation_fit.InterpolationFitter(
        x_aggregate_t, xc=xc_t, nbhr=nbhr, fit_samples=fit_samples, val_samples=val_samples, test_samples=test_samples)
    caliber = nbhr.shape[1]
    error, alpha_opt = fitter.optimized_relative_error(caliber, alpha, return_weights=True)
    # Interpolation validation error = error[:, 1]
    data = error[:, 2:]
    return hm.setup.interpolation.Interpolator(nbhr[:, :caliber], data, nc)


def geometric_neighbors(aggregate_size: int, nc: int):
    """
    Returns the relative indices of the interpolation set of each fine variable in a window. Center neighbors are
    listed first, then neighboring aggregates.

    Args:
        aggregate_size: size of window (aggregate).
        nc: number of coarse variables per aggregate. In 1D at the finest level, we know there are two principal
            components, so nc=2 makes sense there.

    Returns: array of size w x {num_neighbors} of coarse neighbor indices (relative to fine variable indices) of each
        fine variable.
    """
    # Here we assume w points per aggregate and the same number mc of coarse vars per aggregate, but in general
    # aggregate sizes may vary.
    fine_var = np.arange(aggregate_size, dtype=int)

    # Index of neighboring coarse variable. Left neighboring aggregate for points on the left half of the window;
    # right for right.
    coarse_nbhr = np.zeros_like(fine_var)
    left = fine_var < aggregate_size // 2
    right = fine_var >= aggregate_size // 2
    coarse_nbhr[left] = -1
    coarse_nbhr[right] = 1

    # All nbhrs = central aggregate coarse vars + neighboring aggregate coarse vars.
    coarse_vars_center = np.tile(np.arange(nc, dtype=int), (aggregate_size, 1))
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


def update_interpolation(p: scipy.sparse.csr_matrix, e, ec, nbhr: np.ndarray, threshold: float = 0.1) -> \
    scipy.sparse.csr_matrix:
    """
    Updates an interpolation for a new vector using Kaczmarz (minimum move from current interpolation while
    satisfying |e-P*ec| <= t.

    Args:
        p: old interpolation matrix.
        e: new test vector.
        ec: coarse representation of e.
        nbhr: (i, j) index pairs indicating the sparsity pattern of the interpolation update.
        threshold: coarsening accuracy threshold in e.

    Returns:
        updated interpolation. Sparsity pattern = p's pattern union nbhr.
    """
    r = e - p.dot(ec)
    s = np.sign(r)
    # Lagrange multiplier.
    row, col = zip(*sorted(set(zip(i, j)) + set(nbhr[:, 0], nbhr[:, 1])))
    E = ec[row, col]
    lam = (r - s * (t ** 0.5)) / np.sum(E ** 2, axis=1)
    return p + lam * E
