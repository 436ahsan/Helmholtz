"""Interpolation construction routines. Fits interpolation to 1D Helmholtz test functions in particular (with specific
coarse neighborhoods based on domain periodicity)."""
import logging
import numpy as np
import scipy.sparse
from typing import Tuple
from scipy.linalg import norm

import helmholtz as hm

_LOGGER = logging.getLogger(__name__)

_AGGREGATOR = {"mean": np.mean, "max": np.max}


def create_interpolation_least_squares_domain(
        x: np.ndarray,
        a: scipy.sparse.csr_matrix,
        r: scipy.sparse.csr_matrix,
        fine_location: np.ndarray,
        domain_size: float,
        aggregate_size: int = None,
        num_components: int = None,
        neighborhood: str = "extended",
        num_test_examples: int = 5,
        repetitive: bool = False,
        caliber: int = None,
        max_caliber: int = 6,
        num_windows: int = None,
        target_error: float = 0.2,
        kind: str = "l2",
        fit_scheme: str = "ridge",
        weighted: bool = False,
        ritz: bool = False) -> \
        scipy.sparse.csr_matrix:
    """
    Creates the interpolation operator P by least squares fitting from r*x to x. Interpolatory sets are automatically
    determined by a's graph.
    Args:
        x: fine-level test matrix. shape = (num_fine_vars, num_examples).
        a: fine-level operator (only its adjacency graph is used here).
        r: coarsening operator.
        fine_location: fine-level variable location.
        aggregate_size: size of aggregate (assumed uniform over domain).
        num_components: number of coarse variables per aggregate.
        neighborhood: "aggregate"|"extended" coarse neighborhood to interpolate from: only coarse variables in the
            aggregate, or the R*A*R^T sparsity pattern.
        num_test_examples: number of test functions dedicated to testing (do not participate in SVD, LS fit).
        repetitive: whether to exploit problem repetitiveness by creating a constant R stencil on all aggregates
            using windows from a single (or few) test vectors.
        caliber: if None, automatically selects caliber. If non-None, uses this caliber value.
        max_caliber: maximum interpolation caliber to use.
        num_windows: for the repetitive case, custom # windows (samples) to use. If None, default to 4 * max_caliber.
        target_error: target relative interpolation error in norm 'kind'.
        kind: interpolation norm kind ("l2"|"a" = energy norm).
        fit_scheme: whether to use regularized unweighted LS ("ridge") or plain LS ("plain").
        weighted: whether to use residual-weighted LS or not.

    Returns:
        interpolation matrix P.
        relative interpolation A-norm error on the test set.
        optimal caliber.
    """
    # Find nearest neighbors of each fine point in an aggregate.
    if repetitive:
        coarse_location = hm.setup.geometry.coarse_locations(fine_location, aggregate_size, num_components)
        nbhr = hm.setup.geometry.geometric_neighbors_from_locations(
            fine_location, coarse_location, domain_size, aggregate_size)
    else:
        nbhr = _get_neighbor_set(x, a, r, neighborhood)

    if ritz:
        # Apply local Ritz projection (on which sub-domain?) to obtain smoother TVs.
        x, _ = hm.linalg.ritz(x, lambda x: a.dot(x))

    # Prepare fine and coarse test matrices.
    xc = r.dot(x)
    residual = a.dot(x)
    max_caliber = min(max_caliber, max(len(n) for n in nbhr))
    if repetitive and num_windows is None:
        fine_level_size, num_test_functions = x.shape
        num_aggregates = int(np.ceil(fine_level_size / aggregate_size))
        num_windows = np.minimum(num_aggregates * num_test_functions, 12 * max_caliber)
    fitter, fold_sizes = _create_interpolation_fitter(
        x, xc, residual, aggregate_size, num_components, num_windows, repetitive=repetitive,
        num_test_examples=num_test_examples, fit_scheme=fit_scheme, weighted=weighted)
    folds = tuple(f.T for f in hm.linalg.create_folds(x.T, fold_sizes))

    calibers = np.array([caliber]) if caliber is not None else np.arange(1, max_caliber + 1, dtype=int)
    # Increase caliber (fitting interpolation with LS) until the test error A-norm is below the accuracy threshold.
    for caliber in calibers:
        nbhr_caliber = [n[:caliber] for n in nbhr]
        p = fitter(nbhr_caliber)
        if repetitive:
            # TODO(oren): this will not work for the last aggregate if aggregate_size does not divide the domain size.
            p = _tile_interpolation_matrix(p, aggregate_size, num_components, x.shape[0])
        error = dict((kind, np.array([relative_interpolation_error(p, r, a, f, kind) for f in folds]))
                     for kind in ("l2", "a"))
        _LOGGER.debug("caliber {} error l2 {} a {}".format(
            caliber,
            np.array2string(error["l2"], separator=", ", formatter={'float_kind': lambda x: "%4.2f" % x}),
            np.array2string(error["a"], separator=", ", formatter={'float_kind': lambda x: "%.2e" % x})))
        # Check test set error.
        if error[kind][-1] < target_error:
            return p

    if caliber is None:
        _LOGGER.warning("Could not find a good caliber for threshold {}, using max caliber {}".format(
            target_error, caliber))
    return p


def _create_interpolation_fitter(x: np.ndarray, xc: np.ndarray, residual: np.ndarray,
                                 aggregate_size: int,
                                 num_components: int,
                                 num_windows: int,
                                 repetitive: bool = False, num_test_examples: int = 5,
                                 fit_scheme: str = "ridge",
                                 regularize: bool = False,
                                 weighted: bool = False):
    """
    Creates an interpolation factory over the samples: a single aggregate (if repetitive) or entire domain (otherwise).

    :param x:
    :param xc:
    :param residual:
    :param aggregate_size:
    :param num_components:
    :param repetitive:
    :param num_test_examples:
    :param num_windows:
    :param fit_scheme:
    :param weighted:
    :return:
    """
    if repetitive:
        # TODO(orenlivne): change window stride from aggregate_size to nc; we can create more windows this way. xc
        # needs to be locally computed on each window=aggregate in that case.
        # Create windows of 'x'.
        x_disjoint_aggregate_t = hm.linalg.get_windows_by_index(
            x, np.arange(aggregate_size), aggregate_size, num_windows)

        # Create corresponding windows of 'xc'. Note: we are currently including the entire domainin the window.
        # TODO(orenlivne): reduce storage here using smart periodic indexing or calculate the nbhr set here first
        # and only pass windows of nbhr values to create_interpolation.
        xc_disjoint_aggregate_t = hm.linalg.get_windows_by_index(
            xc, np.arange(xc.shape[0]), num_components, num_windows)

        # Create local residual norms corresponding to the 'x'-windows. 3 * aggregate_size domain is good for 1D.
        # TODO(orenlivne): in graph problems, replace residual_window_size by aggregate_size + sum of its
        #  residual_window_size aggregate sizes.
        r_norm_disjoint_aggregate_t = hm.setup.sampling.residual_norm_windows(
            residual, 3 * aggregate_size, aggregate_size, num_windows)
    else:
        x_disjoint_aggregate_t, xc_disjoint_aggregate_t = x.T, xc.T
        # TODO(orenlivne): fix this to be all local residual norms in the non-repetitive case.
        r_norm_disjoint_aggregate_t = None

    if weighted:
        # Weighted LS: sum(w*(xc- x))^2 = sum(w^2*xc^2 - w*x^2).
        weight = np.clip(r_norm_disjoint_aggregate_t, 1e-15, None) ** (-1)
        #print("r_norm_disjoint_aggregate_t", r_norm_disjoint_aggregate_t)
    else:
        weight = np.ones_like(x_disjoint_aggregate_t)

    # Create folds.
    num_examples = int(x_disjoint_aggregate_t.shape[0])
    num_ls_examples = num_examples - num_test_examples
    val_samples = int(0.5 * num_ls_examples)
    fit_samples = num_examples - val_samples - num_test_examples

    if repetitive:
        # Since we are fitting to a sample of windows, there is no sense in calculating the interpolation error of
        # parts of vectors to exactly correspond to the non-repetitive case. Simply lump all test vectors into a single
        # fold here.
        fold_sizes = (int(x.shape[1]),)
    else:
        fold_sizes = (fit_samples, val_samples, num_test_examples)

    # Ridge regularization parameter (list of values).
    if regularize:
        alpha = np.array([0, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    else:
        alpha = np.array([0])
        #alpha = np.logspace(1e-4, 2, 10)

    if fit_scheme == "plain":
        fitter = lambda nbhr: \
            hm.setup.interpolation_ls_fit.create_interpolation_least_squares_plain(
                x_disjoint_aggregate_t, xc_disjoint_aggregate_t, nbhr, weight)
    elif fit_scheme == "ridge":
        fitter = lambda nbhr: \
            hm.setup.interpolation_ls_fit.create_interpolation_least_squares_ridge(
                x_disjoint_aggregate_t, xc_disjoint_aggregate_t, nbhr, weight,
                                                                                   alpha=alpha, fit_samples=fit_samples,
                                                                                   val_samples=val_samples,
                                                                                   test_samples=num_test_examples)
    else:
        raise Exception("Unsupported interpolation fitting schema {}".format(fit_scheme))

    return fitter, fold_sizes

def relative_interpolation_error(p: scipy.sparse.csr_matrix,
                                 r: scipy.sparse.csr_matrix,
                                 a: scipy.sparse.csr_matrix,
                                 x: np.ndarray,
                                 kind: str,
                                 aggregation: str = "mean") -> float:
    """
    Returns the relative interpolation error.

    Args:
        p: interpolation matrix.
        r: coarsening matrix.
        a: fine-level operator.
        x: test matrix.
        kind: kind of norm ("l2" | "a", for the A'*A norm).
        aggregation: how to aggregate the norm across test vectors ("mean" | "max").

    Returns:
        relative interpolation error.
    """
    if kind == "l2":
        error = norm(x - p.dot(r.dot(x)), axis=0) / norm(x, axis=0)
    elif kind == "a":
        error = norm(a.dot(x - p.dot(r.dot(x))), axis=0) / norm(a.dot(x), axis=0)
    else:
        raise Exception("Unsupported error norm {}".format(kind))
    aggregator = _AGGREGATOR[aggregation]
    return aggregator(error)


def sort_neighbors_by_similarity(x_aggregate: np.array, xc: np.array, nbhr: np.array):
    return np.array([nbhr_of_i[
                         np.argsort(-_similarity(x_aggregate[:, i][:, None], xc[:, nbhr_of_i]))
                     ][0] for i, nbhr_of_i in enumerate(nbhr)])


def _get_neighbor_set(x, a, r, neighborhood):
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


def _similarity(x, xc):
    """Returns all pairwise cosine similarities between the two test matrices. Does not zero out the mean, which
    assumes intercept = False in fitting interpolation.
    """
    return hm.linalg.pairwise_cos_similarity(x, xc, squared=True)

# def update_interpolation(p: scipy.sparse.csr_matrix, e, ec, nbhr: np.ndarray, threshold: float = 0.1) -> \
#     scipy.sparse.csr_matrix:
#     """
#     Updates an interpolation for a new vector using Kaczmarz (minimum move from current interpolation while
#     satisfying |e-P*ec| <= t.
#
#     Args:
#         p: old interpolation matrix.
#         e: new test vector.
#         ec: coarse representation of e.
#         nbhr: (i, j) index pairs indicating the sparsity pattern of the interpolation update.
#         threshold: coarsening accuracy threshold in e.
#
#     Returns:
#         updated interpolation. Sparsity pattern = p's pattern union nbhr.
#     """
#     r = e - p.dot(ec)
#     s = np.sign(r)
#     # Lagrange multiplier.
#     row, col = zip(*sorted(set(zip(i, j)) + set(nbhr[:, 0], nbhr[:, 1])))
#     E = ec[row, col]
#     lam = (r - s * (t ** 0.5)) / np.sum(E ** 2, axis=1)
#     return p + lam * E
