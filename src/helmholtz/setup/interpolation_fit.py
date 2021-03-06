"""Fits interpolation to test functions from nearest coarse neighbors using regularized least-squares. This is a
generic module that does not depend on Helmholtz details."""
import numpy as np
import scipy.linalg
import sklearn.metrics.pairwise
from numpy.linalg import norm
from typing import List, Tuple

_SMALL_NUMBER = 1e-15


class InterpolationFitter:
    """
    Fits interpolation from X[:, n[0]],...,X[:, n[k-1]], i's k nearest neighbors, to X[:, i] using
    Ridge regression. Determines the optimal Ridge parameter by cross-validation.
    """

    def __init__(self,
                 x: np.ndarray,
                 xc: np.ndarray = None,
                 nbhr: np.ndarray = None,
                 fit_samples: int = 1000,
                 val_samples: int = 1000,
                 test_samples: int = 1000):
        """Creates an interpolation fitter from XC[:, n[0]],...,XC[:, n[k-1]], i's k nearest neighbors,
        to Y[:, i], for each i, using Ridge regression .

        Args:
            x: activation matrix to fit interpolation to ("fine variables").
            xc: activation matrix to fit interpolation from ("coarse variables"). If None, XC=X.
            nbhr: nearest neighbor of each activation (nbhr[i] = sorted neighbors by descending proximity). If None,
                calculated inside this object based on Pearson correlation distance.
            fit_samples: number of samples to use for fitting interpolation.
            val_samples: number of samples to use for determining interpolation fitting regularization parameter.
            test_samples: number of samples to use for testing interpolation generalization.
        """
        self._x = x
        self._xc = xc if xc is not None else x
        self._fit_samples = fit_samples
        self._val_samples = val_samples
        self._test_samples = test_samples
        if nbhr is None:
            self._similarity = None
            # Calculate all xc neighbors of each x activation.
            self._nbhr = np.argsort(-self.similarity, axis=1)
        else:
            self._nbhr = nbhr

    @property
    def similarity(self):
        """Returns all pairwise correlation similarities between x and xc. Cached.

        ***DOES NOT ZERO OUT MEAN, which assumes intercept = False in fitting interpolation.***
        """
        # Calculate all pairwise correlation similarities between x and xc. Zero out mean here.
        if self._similarity is None:
            # If we allow an affine interpolation (constant term), then subtract the mean here.
            x, xc = self._x, self._xc
            #            x = self._x - np.mean(self._x, axis=0)
            #            xc = self._xc - np.mean(self._xc, axis=0)
            self._similarity = sklearn.metrics.pairwise.cosine_similarity(x.transpose(), xc.transpose())
        return self._similarity

    def relative_error(self, k, alpha_values, intercept: bool = False, test: bool = False,
                       return_weights: bool = False):
        """
        Returns the fit and test/validation set relative interpolation error for a list of regularization parameter
        values.
        Args:
            k: interpolation caliber.
            alpha_values: list of alpha values to calculate the interpolation error for.
            intercept: whether to add an affine term to the interpolation.
            test: whether to use the test set (iff True) or validation set.
            return_weights: iff True, also returns the weights in the result array.

        Returns: relative interpolation error: array, shape=(2 + return_weights * (k + intercept), num_activations,
            len(alpha_values))
        """
        fit_range = (0, self._fit_samples)
        test_range = (self._fit_samples + self._val_samples, self._fit_samples + self._val_samples + self._test_samples) \
            if test else (self._fit_samples, self._fit_samples + self._val_samples)
        xc_fit = self._xc[fit_range[0]:fit_range[1]]
        x_fit = self._x[fit_range[0]:fit_range[1]]
        xc_val = self._xc[test_range[0]:test_range[1]]
        x_val = self._x[test_range[0]:test_range[1]]

        def _interpolation_error(i):
            """Returns the fitting and validation errors from k nearest neighbors (the last two columns returned from
            fit_interpolation()."""
            # Exclude self from k nearest neighbors.
            neighbors = self._nbhr[i, :k]
            return fit_interpolation(xc_fit[:, neighbors], x_fit[:, i],
                                     xc_val[:, neighbors], x_val[:, i],
                                     alpha_values, intercept=intercept, return_weights=return_weights)

        # TODO(oren): parallelize this if we fit interpolation to many fine variables.
        error = [_interpolation_error(i) for i in self._fine_vars()]
        return np.array(error)

    def optimized_relative_error(self, k, alpha_values, intercept: bool = False,
                                 return_weights: bool = False):
        """Returns the relative interpolation error of each activation on the fitting and test set,
        and optimal alpha. The optimal alpha is determined by minimizing the validation interpolation error over
        the range of values 'alpha_values'.

        Args:
            k: interpolation caliber.
            alpha_values: list of alpha values to optimize over.
            intercept: whether to add an affine term to the interpolation.
            return_weights: iff True, also returns the weights in the result array.

        Returns: relative interpolation error: array, shape=(num_activations, len(alpha_values))
        """
        # Calculate interpolation error vs. alpha for each k for fitting, validation sets.
        error = self.relative_error(k, alpha_values, intercept=intercept)
        # Minimize validation error.
        error_val = error[:, :, 1]
        alpha_opt_index = np.argmin([_filtered_mean(error_val[:, j]) for j in range(error_val.shape[1])])
        alpha_opt = alpha_values[alpha_opt_index]
        # Recalculate interpolation (not really necessary if we store it in advance) for the optimal alpha and
        # calculate the fit and test set interpolation errors.
        return self.relative_error(
            k, alpha_opt, intercept=intercept, return_weights=return_weights, test=True), alpha_opt

    def _fine_vars(self):
        return range(self._x.shape[1])


def fit_interpolation(xc_fit, x_fit, xc_val, x_val, alpha, intercept: bool = False, return_weights: bool = False):
    """
    Fits interpolation from xc_fit to x_fit using Ridge regression.

    Args:
        xc_fit: coarse variable matrix of the k nearest neighbors of x_fit to fit interpolation from. Each column is a
          coarse variable. Each row is a sample. Note: must have more rows than columns for an over-determined
          least-squares.
        x_fit: fine variable vector to fit interpolation to. Each column is a fine variable. Row are samples.
        xc_val: coarse activation matrix of validation samples.
        x_val: fine variable vector of validation samples.
        alpha: Ridge regularization parameter (scalar or list of values).
        intercept: whether to add an intercept or not.
        return_weights: if True, returns the interpolation coefficients + errors. Otherwise, just errors.

        info: len(a) x (2 + (k + intercept) * return_weights) matrix containing the [interpolation coefficients and]
        relative interpolation error on fitting samples and validation samples or each value in alpha. If
        intercept = True, its coefficient is info[:, 2], and subsequent columns correspond to the nearest xc neighbors
        in order of descending proximity.
    """
    if intercept:
        xc_fit = np.concatenate((np.ones((xc_fit.shape[0], 1)), xc_fit), axis=1)
    m, n = xc_fit.shape
    assert m > n, "Number of samples ({}) must be > number of variables ({}) for LS fitting.".format(m, n)
    x_fit_norm = norm(x_fit)

    # The SVD computation part that does not depend on alpha.
    u, s, vt = scipy.linalg.svd(xc_fit)
    v = vt.transpose()
    q = s * (u.transpose()[:n].dot(x_fit))

    # Validation quantities.
    if intercept:
        xc_val = np.concatenate((np.ones((xc_val.shape[0], 1)), xc_val), axis=1)
    x_val_norm = norm(x_val)

    def _solution_and_errors(a):
        p = v.dot((q / (s ** 2 + a * x_fit_norm ** 2)))
        info = [norm(xc_fit.dot(p) - x_fit) / np.clip(x_fit_norm, _SMALL_NUMBER, None),
                norm(xc_val.dot(p) - x_val) / np.clip(x_val_norm, _SMALL_NUMBER, None)]
        if return_weights:
            info += list(p)
        return info

    return np.array([_solution_and_errors(a) for a in alpha]) \
        if isinstance(alpha, (list, np.ndarray)) else np.array(_solution_and_errors(alpha))


def optimized_fit_interpolation(xc_fit, x_fit, xc_val, x_val, alpha: np.ndarray, intercept: bool = False,
                                return_weights: bool = False) -> Tuple[float, np.ndarray]:
    """
    Fits interpolation from xc_fit to x_fit using Ridge regression and chooses the optimal regularization parameter
    to minimize validation fit error.

    Args:
        xc_fit: coarse variable matrix of the k nearest neighbors of x_fit to fit interpolation from. Each column is a
          coarse variable. Each row is a sample. Note: must have more rows than columns for an over-determined
          least-squares.
        x_fit: fine variable vector to fit interpolation to. Each column is a fine variable. Row are samples.
        xc_val: coarse activation matrix of validation samples.
        x_val: fine variable vector of validation samples.
        alpha: Ridge regularization parameter (list of values).
        intercept: whether to add an intercept or not.
        return_weights: if True, returns the interpolation coefficients + errors. Otherwise, just errors.

    Retuns: Tuple (alpha, info)
        alpha: optimal regularization parameter
        info: (2 + (k + intercept) * return_weights) vector containing the [interpolation coefficients and]
        relative interpolation error on fitting samples and validation samples or each value in alpha. If
        intercept = True, its coefficient is info[:, 2], and subsequent columns correspond to the nearest xc neighbors
        in order of descending proximity.
    """
    info = fit_interpolation(xc_fit, x_fit, xc_val, x_val, alpha, intercept=intercept, return_weights=return_weights)
    # Minimize validation error.
    alpha_opt_index = np.argmin(info[:, 1])
    return alpha[alpha_opt_index], info[alpha_opt_index]


def create_interpolation_least_squares(
        x: np.ndarray,
        xc: np.ndarray,
        nbhr: List[np.ndarray],
        alpha: np.ndarray,
        fit_samples: int = None,
        val_samples: int = None,
        test_samples: int = None) -> Tuple[scipy.sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates the next coarse level's R and P operators.
    Args:
        x: fine-level test matrix.
        xc: coarse-level test matrix.
        nbhr: list of neighbor lists for all fine points.
        alpha: Ridge regularization parameter (list of values).
        fit_samples: number of samples to use for fitting interpolation.
        val_samples: number of samples to use for determining interpolation fitting regularization parameter.
        test_samples: number of samples to use for testing interpolation generalization.

    Returns:
        interpolation matrix P,
        relative fit error at all fine points,
        relative validation error at all fine points,
        relative test error at all fine points,
        optimal alpha for all fine points.
    """
    # Divide into folds.
    num_examples, n = x.shape
    assert len(nbhr) == n
    nc = xc.shape[1]
    if fit_samples is None or val_samples is None or test_samples is None:
        fit_samples, val_samples, test_samples = num_examples // 3, num_examples // 3, num_examples // 3
    fit_range = (0, fit_samples)
    val_range = (fit_samples, fit_samples + val_samples)
    test_range = (fit_samples + val_samples, fit_samples + val_samples + test_samples)
    xc_fit  = xc[fit_range [0]:fit_range [1]]
    x_fit   = x [fit_range [0]:fit_range [1]]
    xc_val  = xc[val_range [0]:val_range [1]]
    x_val   = x [val_range [0]:val_range [1]]
    xc_test = xc[test_range[0]:test_range[1]]
    x_test  = x [test_range[0]:test_range[1]]

    # Fit interpolation by least-squares.
    result = [optimized_fit_interpolation(xc_fit[:, nbhr_i], x_fit[:, i], xc_val[:, nbhr_i], x_val[:, i],
                                          alpha, return_weights=True)
              for i, nbhr_i in enumerate(nbhr)]
    alpha_opt = np.array([row[0] for row in result])
    info = [row[1] for row in result]
    # In each info element:
    # Interpolation fit error = error[:, 0]
    # Interpolation validation error = error[:, 1]
    # Interpolation coefficients = error[:, 2:]
    fit_error = np.array([info_i[0] for info_i in info])
    val_error = np.array([info_i[1] for info_i in info])

    # Build the sparse interpolation matrix.
    row = np.concatenate(tuple([i] * len(nbhr_i) for i, nbhr_i in enumerate(nbhr)))
    col = np.concatenate(tuple(nbhr))
    data = np.concatenate(tuple(info_i[2:] for info_i in info))
    p = scipy.sparse.coo_matrix((data, (row, col)), shape=(n, nc)).tocsr()

    return p, fit_error, val_error, relative_interpolation_error(p, x_test, xc_test), alpha_opt


def _filtered_mean(x):
    """Returns the mean of x except large outliers."""
    return x[x <= 3 * np.median(x)].mean()


def relative_interpolation_error(p: scipy.sparse.csr_matrix, x: np.ndarray, xc: np.ndarray):
    """Returns the relative interpolation error (error norm is L2 norm over all test vectors, for each fine
    point)."""
    px = p.dot(xc.transpose()).transpose()
    if x.ndim == 1:
        return (x - px) / x
    else:
        return norm(x - px, axis=0) / norm(x, axis=0)


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
