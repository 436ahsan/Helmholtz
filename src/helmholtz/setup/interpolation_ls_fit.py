"""Fits interpolation to test functions from nearest coarse neighbors using regularized least-squares. This is a
generic module that does not depend on Helmholtz details."""
import numpy as np
import scipy.linalg
import sklearn.metrics.pairwise
from numpy.linalg import norm
from typing import List, Tuple

import helmholtz as hm

_SMALL_NUMBER = 1e-15


class InterpolationLsFitter:
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
            self._similarity = sklearn.metrics.pairwise.cosine_similarity(x.T, xc.T)
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
        folds = (self._fit_samples, self._val_samples, self._test_samples)
        x_fit, x_val, x_test = hm.linalg.create_folds(x, folds)
        xc_fit, xc_val, xc_test = hm.linalg.create_folds(x, folds)
        if test:
            x_val = x_test
            xc_val = xc_test

        def _interpolation_error(i):
            """Returns the fitting and validation errors from k nearest neighbors (the last two columns returned from
            fit_interpolation()."""
            # Exclude self from k nearest neighbors.
            neighbors = self._nbhr[i][:k]
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
    v = vt.T
    q = s * (u.T[:n].dot(x_fit))

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

    # print(x_fit)
    # print(xc_fit)
    # print("p", _solution_and_errors(0))
    # print("ls", np.linalg.lstsq(xc_fit, x_fit))

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


def create_interpolation_least_squares_ridge(
        x: np.ndarray,
        xc: np.ndarray,
        nbhr: List[np.ndarray],
        weight: np.ndarray,
        alpha: np.ndarray,
        fit_samples: int = None,
        val_samples: int = None,
        test_samples: int = None) -> scipy.sparse.csr_matrix:
    """
    Creates the next coarse level interpolation operator P using ridge-regularized, ueighted least-squares.
    Args:
        x: fine-level test matrix.
        xc: coarse-level test matrix.
        nbhr: list of neighbor lists for all fine points.
        weight: least-squares weights to apply (same size as x).
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
        fit_samples, val_samples, test_samples = num_examples // 2, num_examples // 4
        test_samples = num_examples - fit_samples - val_samples
    folds = (fit_samples, val_samples, test_samples)
    x_fit, x_val, x_test = hm.linalg.create_folds(x, folds)
    xc_fit, xc_val, xc_test = hm.linalg.create_folds(xc, folds)
    weight_fit, weight_val, _ = hm.linalg.create_folds(weight, folds)

    #print(alpha, x_fit.shape, x_val.shape)
    # Fit interpolation by least-squares.
    i = 0
    result = [optimized_fit_interpolation(
        np.diag(weight_fit[:, i]).dot(xc_fit[:, nbhr_i]),
        np.diag(weight_fit[:, i]).dot(x_fit[:, i]),
        np.diag(weight_val[:, i]).dot(xc_val[:, nbhr_i]),
        np.diag(weight_val[:, i]).dot(x_val[:, i]),
        alpha, return_weights=True)
        for i, nbhr_i in enumerate(nbhr)]
    info = [row[1] for row in result]
    # In each 'info' element:
    # Interpolation fit error = error[:, 0]
    # Interpolation validation error = error[:, 1]
    # Interpolation coefficients = error[:, 2:]
    # fit_error = np.array([info_i[0] for info_i in info])
    # val_error = np.array([info_i[1] for info_i in info])
    # alpha_opt = np.array([row[0] for row in result])
    p_coefficients = [info_i[2:] for info_i in info]

    return _create_csr_matrix(nbhr, p_coefficients, nc)


def create_interpolation_least_squares_plain(
        x: np.ndarray,
        xc: np.ndarray,
        nbhr: List[np.ndarray],
        weight: np.ndarray) -> scipy.sparse.csr_matrix:
    """
    Creates the next coarse level interpolation operator P using plain weighted least-squares on a fitting set.

    Args:
        x: fine-level test matrix - fit set.
        xc: coarse-level test matrix - fit set.
        nbhr: list of neighbor lists for all fine points.
        weight: least-squares weights to apply (same size as x).

    Returns:
        interpolation matrix P,
        relative fit error at all fine points,
        relative validation error at all fine points,
        relative test error at all fine points,
        optimal alpha for all fine points.
    """
    n = x.shape[1]
    nc = xc.shape[1]
    assert len(nbhr) == n
    #print("plain LS {}".format(x.shape))
    # Fit interpolation by unregularized weighted least-squares.
    p_coefficients = [np.linalg.lstsq(
        np.diag(weight[:, i]).dot(xc[:, nbhr_i]),
        np.diag(weight[:, i]).dot(x[:, i]),
        rcond=None
    )[0] for i, nbhr_i in enumerate(nbhr)]
    return _create_csr_matrix(nbhr, p_coefficients, nc)


def _create_csr_matrix(nbhr, p_coefficients, nc):
    # Build the sparse interpolation matrix.
    n = len(nbhr)
    row = np.concatenate(tuple([i] * len(nbhr_i) for i, nbhr_i in enumerate(nbhr)))
    col = np.concatenate(tuple(nbhr))
    data = np.concatenate(tuple(p_coefficients))
    return scipy.sparse.coo_matrix((data, (row, col)), shape=(n, nc)).tocsr()


def _filtered_mean(x):
    """Returns the mean of x except large outliers."""
    return x[x <= 3 * np.median(x)].mean()
