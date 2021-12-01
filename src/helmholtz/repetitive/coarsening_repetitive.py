"""Repetitive coarsening (R) construction routines."""
import logging
import numpy as np
import scipy.sparse
from numpy.linalg import svd
from typing import List, Tuple

import helmholtz as hm

_LOGGER = logging.getLogger(__name__)


class Coarsener:
    """
    Encapsulates the restriction operator as both a full array over an aggregate (for easy tiling) and sparse CSR matrix
    format. Assumes non-overlapping aggregate (block sparsity) structure."""

    def __init__(self, r: np.ndarray):
        """
        Creates an interpolation into a window (aggregate).
        Args:
            r: {aggregate_size} x {aggregate_size} coarse variable definition over an aggregate. Includes all possible
                coarse vars, out of which we select nc based on an energy threshold in tile().
        """
        # Convert matrix to array if needed.
        self._r = np.array(r)

    def asarray(self) -> np.ndarray:
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


def create_coarsening(x_aggregate_t, threshold: float, num_components: int = None, normalize: bool = False) -> \
        Tuple[Coarsener, np.ndarray]:
    """
    Generates R (coarse variables) on an aggregate from SVD principal components.

    Args:
        x_aggregate_t: fine-level test matrix on an aggregate, transposed.
        threshold: relative reconstruction error threshold. Determines nc.
        num_components: if not None, overrides threshold with this fixed number of principal components.
        normalize: if True, scales the row sums of R to 1.

    Returns:
        coarsening operator nc x {aggregate_size}, list of all singular values on aggregate.
    """
    u, s, vh = svd(x_aggregate_t)
    num_components = num_components if num_components is not None else \
        _get_interpolation_caliber(s, np.array([threshold]))[0]
    r = vh[:num_components]
    if normalize:
        r /= r.sum(axis=1)
    return Coarsener(r), s


def create_coarsening_domain(x, threshold: float = 0.1, max_coarsening_ratio: float = 0.5,
                             max_aggregate_size: int = 8, fixed_aggregate_size: int = None) -> \
        Tuple[scipy.sparse.csr_matrix, List[np.ndarray]]:
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
    nc = []
    energy_error = []
    while start < x.shape[0]:
        r, e = _create_aggregate_coarsening(x, threshold, max_coarsening_ratio, max_aggregate_size, start,
                                            fixed_aggregate_size=fixed_aggregate_size)
        r_aggregate.append(r)
        energy_error.append(e)
        aggregate_size = r.shape[1]
        aggregates.append(np.arange(start, start + aggregate_size))
        nc.append(r.shape[0])
        start += aggregate_size

    # Merge all aggregate coarsening operators.
    return scipy.sparse.block_diag(r_aggregate).tocsr(), aggregates, np.array(nc), energy_error


def _create_aggregate_coarsening(x, threshold, max_coarsening_ratio, max_aggregate_size, start,
                                 fixed_aggregate_size: int = None):
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
    energy_error = 1
    # TODO(orenlivne): find a better strategy to locate aggregate size that does not involve incremental strategy
    # (which is a lot more expensive than multiplicative).
    # while (aggregate_size <= max_aggregate_size // aggregate_expansion_factor) and (end < x.shape[0]) and \
    #         (coarsening_ratio > max_coarsening_ratio):
        #aggregate_size *= aggregate_expansion_factor
    if fixed_aggregate_size is not None:
        aggregate_size = fixed_aggregate_size
    while (aggregate_size < max_aggregate_size) and (end < x.shape[0]) and \
            (coarsening_ratio > max_coarsening_ratio):
        if fixed_aggregate_size is None:
            aggregate_size += 1
        end = min(start + aggregate_size, domain_size)
        x_aggregate_t = x[start:end].transpose()
        r, s = create_coarsening(x_aggregate_t, threshold)
        r = r.asarray()
        # n = actual aggregate size after trimming to domain end. nc = #coarse variables.
        nc, n = r.shape
        coarsening_by_aggregate_size[n] = r
        coarsening_ratio = nc / n
        energy_error = (sum(s[nc:] ** 2) / sum(s ** 2)) ** 0.5
        _LOGGER.debug("SVD {:2d} x {:2d} nc {} cr {:.2f} err {:.3f} sigma {}"
                      " err {}".format(x_aggregate_t.shape[0], x_aggregate_t.shape[1], nc, coarsening_ratio,
                                       energy_error, np.array2string(s, separator=", ", precision=2),
                                       np.array2string(
                                             (1 - np.cumsum(s ** 2) / sum(s ** 2)) ** 0.5, separator=", ",
                                             precision=2)))
        if fixed_aggregate_size is not None:
            break
    r, n = min(((r, n) for n, r in coarsening_by_aggregate_size.items()), key=lambda item: item[0].shape[0] / item[1])
    if r.shape[0] == aggregate_size:
        energy_error = 0
    if r.shape[0] / aggregate_size > max_coarsening_ratio and fixed_aggregate_size is None:
        _LOGGER.warning("Could not find a good coarsening ratio for aggregate {}:{}, n {} nc {} cr {:.2f}".format(
            start, start + n, n, r.shape[0], r.shape[0] / n
        ))
    return r, energy_error


def create_ptaq_matrix(fine_level: hm.hierarchy.multilevel.Level,
                    fine_location: np.ndarray,
                    x: np.ndarray,
                    coarse_level: hm.hierarchy.multilevel.Level,
                    coarse_location: np.ndarray,
                    aggregate_size: int,
                    num_components: int,
                    caliber: int,
                    max_caliber: int = 6,
                    num_test_examples: int = 5):
    # TODO(orenlivne): convert everything below to local computations, since it's a repetitive framework.
    pta = coarse_level.p.transpose().dot(fine_level.a)
    # Arbitrarily use the first point in the domain as the center of P^T*A.
    # TODO(orenlivne): should that be replaced by the union of P^T*A stencils for i=0..num_components-1? We know
    # it won't make a difference at level 0 but maybe at level 1 or in more general problems?
    i = 0
    pta_vars = np.sort(pta[i].nonzero()[1])

    # Calculate 'rap_vars' = target sparsity pattern = P^T*A*R^T sparsity pattern.
    rap = pta.dot(coarse_level.r.transpose())
    rap_vars = np.sort(rap[i].nonzero()[1])

    q = _create_q_interpolation_matrix(fine_level, fine_location, x, coarse_level, coarse_location,
                                       max_caliber, aggregate_size, num_components, caliber, pta_vars, rap_vars,
                                       num_test_examples=num_test_examples)
    ptaq = pta[:num_components, pta_vars].dot(q)
    return _tile_csr_matrix(ptaq, rap_vars, coarse_level.size)


def create_qtap_matrix(fine_level: hm.hierarchy.multilevel.Level,
                    fine_location: np.ndarray,
                    x: np.ndarray,
                    coarse_level: hm.hierarchy.multilevel.Level,
                    coarse_location: np.ndarray,
                    aggregate_size: int,
                    num_components: int,
                    caliber: int,
                    max_caliber: int = 6,
                    num_test_examples: int = 5):
    # TODO(orenlivne): convert everything below to local computations, since it's a repetitive framework.
    ap = fine_level.a.dot(coarse_level.p)
    # Arbitrarily use the first aggregate in the domain as the center of P^T*A.
    pt = coarse_level.p.transpose()
    pt_vars = np.unique(pt[:num_components].nonzero()[1])

    # Calculate 'rap_vars' = target sparsity pattern = P^T*A*R^T sparsity pattern.
    rap = coarse_level.r.dot(ap)
    rap_vars = np.unique(rap[:num_components].nonzero()[1])

    q = _create_q_interpolation_matrix(fine_level, fine_location, x, coarse_level, coarse_location,
                                       max_caliber, aggregate_size, num_components, caliber, pt_vars, rap_vars,
                                       num_test_examples=num_test_examples)
    qtap = q.transpose().dot(ap[:num_components, rap_vars])
    return _tile_csr_matrix(qtap, rap_vars, coarse_level.size)


def _create_q_interpolation_matrix(fine_level, fine_location, x, coarse_level, coarse_location, max_caliber,
                                   aggregate_size, num_components, caliber, pta_vars, rap_vars,
                                   num_test_examples: int = 5):
    # Find P^T*A fine and coarse var locations. Wrapping works on locations a well even
    # though they are not integers (apply periodic B.C. so distances don't wrap around).
    n = fine_level.size
    location_f = hm.setup.sampling.wrap_index_to_low_value(fine_location[pta_vars], n)
    location_c = hm.setup.sampling.wrap_index_to_low_value(coarse_location[rap_vars], n)
    # Find nearest neighbors of each fine P^T*A point (pta_vars).
    # These are INDICES into the rap_vars array.
    nbhr = np.argsort(np.abs(location_f[:, None] - location_c), axis=1)
    num_aggregates = int(np.ceil(fine_level.size / aggregate_size))
    # Prepare fine and coarse test matrices.
    xc = coarse_level.coarsen(x)
    residual = fine_level.a.dot(x)
    # Prepare samples (windows) of x, xc, residual norm over the local neighborhood only (P^T*A / R*A*P).
    domain_size, num_test_functions = x.shape
    max_caliber = min(max_caliber, max(len(n) for n in nbhr))
    num_windows = max(np.minimum(num_aggregates, (12 * max_caliber) // num_test_functions), 1)
    x_disjoint_aggregate_t = hm.setup.sampling.get_windows_by_index(x, pta_vars, aggregate_size, num_windows)
    xc_disjoint_aggregate_t = hm.setup.sampling.get_windows_by_index(xc, rap_vars, num_components, num_windows)
    # Calculate residual norms.
    index = pta_vars
    stride = aggregate_size
    residual_window_size = 3 * aggregate_size  # Good for 1D.
    residual_window_offset = -(residual_window_size // 2)
    r_norm_disjoint_aggregate_t = np.concatenate(tuple(
        np.linalg.norm(
            residual[int(np.rint(np.mean(
                hm.setup.sampling.wrap_index_to_low_value(index + offset, n)
            ))) + residual_window_offset +
                     np.arange(residual_window_size) % residual.shape[0]],
            axis=0)
        for offset in range(0, num_windows * stride, stride))) / residual_window_size ** 0.5
    # In principle, each point in the aggregate should have a slightly shifted residual window, but we just take the
    # same residual norm for all points for simplicity. Should not matter much.
    r_norm_disjoint_aggregate_t = np.tile(r_norm_disjoint_aggregate_t[:, None], (len(index),))
    weight = np.clip(r_norm_disjoint_aggregate_t, 1e-15, None) ** (-1)
    nbhr_for_caliber = [n[:caliber] for n in nbhr]
    # print("nbhr_for_caliber", nbhr_for_caliber)
    # Create folds.
    num_samples = int(x_disjoint_aggregate_t.shape[0])
    num_ls_examples = num_samples - num_test_examples
    val_samples = int(0.2 * num_ls_examples)
    fit_samples = num_samples - val_samples - num_test_examples
    # Ridge regularization parameter (list of values).
    alpha = np.array([0, 0.01, 0.1, 0.1, 1])
    q_local = hm.setup.interpolation_ls_fit.create_interpolation_least_squares_ridge(
        x_disjoint_aggregate_t, xc_disjoint_aggregate_t, nbhr_for_caliber, weight,
        alpha=alpha, fit_samples=fit_samples,
        val_samples=val_samples, test_samples=num_test_examples)
    qq = scipy.sparse.lil_matrix((len(pta_vars), coarse_level.size))
    qq[:, rap_vars] = q_local
    return q_local


def _tile_csr_matrix(ptaq, rap_vars, nc):
    num_components = ptaq.shape[0]
    ac_local = scipy.sparse.lil_matrix((num_components, nc))
    ac_local[:, rap_vars] = ptaq
    ac_local = ac_local.tocsr()
    row_local, col_local = ac_local.nonzero()
    data_local = ac_local.data
    row = (row_local + np.arange(0, nc, num_components)[:, None]).flatten()
    col = (col_local + np.arange(0, nc, num_components)[:, None]).flatten() % nc
    data = np.tile(data_local, nc // num_components)
    ac = scipy.sparse.csr_matrix((data, (row, col)), shape=(nc, nc))
    return ac


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
