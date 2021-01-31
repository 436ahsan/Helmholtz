"""Fits interpolation to test functions from nearest coarse neighbors using regularized least-squares. This is a
generic module that does not depend on Helmholtz details."""
import numpy as np
import scipy.linalg
import sklearn.metrics.pairwise
from numpy.linalg import norm

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
                                     alpha_values,  intercept=intercept, return_weights=return_weights)
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


def _filtered_mean(x):
    """Returns the mean of x except large outliers."""
    return x[x <= 3 * np.median(x)].mean()