import helmholtz as hm
import numpy as np
import pytest
import time
from numpy.linalg import norm, lstsq


class TestInterpolation:

    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.random.seed(0)

    def test_fit_interpolation(self):
        """Tests bare-bones interpolation with given neighborhoods and ridge coefficient alpha."""
        x_fit = np.random.random((10, 6))
        x_val = np.random.random((10, 6))
        nbhr = np.array([[1, 2, 3, 4, 5]])
        i = 0
        k = 3
        alpha = [0, 0.1, 0.2, 1.0]

        xc_fit = 2 * x_fit
        xc_val = 2 * x_val
        info = hm.interpolation_fit.fit_interpolation(xc_fit[:, nbhr[i, :k]], x_fit[:, i],
                                                      xc_val[:, nbhr[i, :k]], x_val[:, i],
                                                      alpha, intercept=True, return_weights=True)

        # Calculate p by a different way - normal equations, and compare.
        for ind, a in enumerate(alpha):
            j = nbhr[i][:k]
            xj = xc_fit[:, j]
            xj = np.concatenate((np.ones((xj.shape[0], 1)), xj), axis=1)
            xi = x_fit[:, i]
            xi_norm = norm(xi)
            p_normal = lstsq(xj.transpose().dot(xj) + a * xi_norm ** 2 * np.eye(xj.shape[1]),
                             xj.transpose().dot(xi), rcond=None)[0]
            err_fit = norm(xj.dot(p_normal) - xi) / xi_norm

            xj_val = xc_val[:, j]
            xj_val = np.concatenate((np.ones((xj_val.shape[0], 1)), xj_val), axis=1)
            err_val = norm(xj_val.dot(p_normal) - x_val[:, i]) / norm(x_val[:, i])

            assert norm(info[ind, 2:] - p_normal) < 1e-8
            assert info[ind, 0] == pytest.approx(err_fit, 1e-8)
            assert info[ind, 1] == pytest.approx(err_val, 1e-8)

    def test_relative_error(self):
        x = np.random.random((20, 6))
        alpha = [0, 0.1, 0.2, 1.0]

        fitter = hm.interpolation_fit.InterpolationFitter(x, fit_samples=10, val_samples=10)
        error = fitter.relative_error(3, alpha, intercept=True)

        assert error.shape == (6, 4, 2)

    def test_optimized_relative_error(self):
        n = 100
        x = np.random.random((300, n))
        k_values = [2, 3, 5, 10, 12]
        alpha = np.array([0, 0.1, 0.2, 1.0])
        expected_alpha_opt_values = [0.0, 0.1, 0.1, 0.1, 0.2]
        for k, expected_alpha_opt in zip(k_values, expected_alpha_opt_values):
            fitter = hm.interpolation_fit.InterpolationFitter(
                x, xc=x[:, :50] + np.random.random((300, 50)),
                fit_samples=100, val_samples=100, test_samples=100)
            error, alpha_opt = fitter.optimized_relative_error(k, alpha, intercept=True)
            assert error.shape == (n, 2)
            assert alpha_opt == pytest.approx(expected_alpha_opt, 1e-8)
