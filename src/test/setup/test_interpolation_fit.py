import numpy as np
import pytest
from numpy.linalg import norm, lstsq
from numpy.ma.testutils import assert_array_almost_equal

import helmholtz as hm


class TestInterpolationFit:

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
        info = hm.setup.interpolation_fit.fit_interpolation(xc_fit[:, nbhr[i, :k]], x_fit[:, i],
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

        fitter = hm.setup.interpolation_fit.InterpolationFitter(x, fit_samples=10, val_samples=10)
        error = fitter.relative_error(3, alpha, intercept=True)

        assert error.shape == (6, 4, 2)

    def test_optimized_relative_error(self):
        n = 100
        x = np.random.random((300, n))
        k_values = [2, 3, 5, 10, 12]
        alpha = np.array([0, 0.1, 0.2, 1.0])
        expected_alpha_opt_values = [0.0, 0.1, 0.1, 0.1, 0.2]
        for k, expected_alpha_opt in zip(k_values, expected_alpha_opt_values):
            fitter = hm.setup.interpolation_fit.InterpolationFitter(
                x, xc=x[:, :50] + np.random.random((300, 50)),
                fit_samples=100, val_samples=100, test_samples=100)
            error, alpha_opt = fitter.optimized_relative_error(k, alpha, intercept=True)
            assert error.shape == (n, 2)
            assert alpha_opt == pytest.approx(expected_alpha_opt, 1e-8)

    def test_optimized_fit_interpolation(self):
        """Tests bare-bones interpolation with given neighborhoods; optimizes the ridge coefficient alpha."""
        x_fit = np.random.random((10, 6))
        x_val = np.random.random((10, 6))
        nbhr = np.array([[1, 2, 3, 4, 5]])
        i = 0
        k = 3
        alpha = [0, 0.1, 0.2, 1.0]

        xc_fit = 2 * x_fit + 0.1 * np.random.random((10, 6)) + 1
        xc_val = 2 * x_val + 0.1 * np.random.random((10, 6)) + 1
        alpha_opt, info = hm.setup.interpolation_fit.optimized_fit_interpolation(
            xc_fit[:, nbhr[i, :k]], x_fit[:, i], xc_val[:, nbhr[i, :k]], x_val[:, i],
            alpha, intercept=True, return_weights=True)

        assert alpha_opt == pytest.approx(0.1, 1e-8)
        assert_array_almost_equal(info, [0.353209,  0.430609, -0.02109 ,  0.061041,  0.161995, -0.013169])

        # Prune small weights.
        weight = info[3:]
        large = np.where(np.abs(weight) >= 0.02)[0]
        info = hm.setup.interpolation_fit.fit_interpolation(xc_fit[:, nbhr[i, large]], x_fit[:, i],
                                                      xc_val[:, nbhr[i, large]], x_val[:, i],
                                                      alpha_opt, intercept=True, return_weights=True)
        assert_array_almost_equal(info, [0.353209,  0.430609, -0.02109 ,  0.061041,  0.161995, -0.013169])

    def test_create_interpolation_least_squares(self):
        n = 10
        num_examples = 30
        x = np.random.random((num_examples, n))
        nc = n // 2
        xc = 2 * x[:, ::2] + 0.1 * np.random.random((num_examples, nc)) + 1
        fine_vars = np.arange(n)
        coarse_vars = np.arange(nc)
        nbhr = [np.take(coarse_vars, np.arange(i - 2, i + (2 if i % 2 == 0 else 1)), mode="wrap") for i in fine_vars]
        alpha = [0, 0.1, 0.2, 1.0]
        p, fit_error, val_error, test_error, alpha_opt = \
            hm.setup.interpolation_fit.create_interpolation_least_squares(x, xc, nbhr, alpha)

        assert_array_almost_equal(alpha_opt, [0.1, 1. , 0.1, 1. , 0. , 1. , 1. , 1. , 0. , 1. ])
        assert_array_almost_equal(fit_error, [0.074688, 0.290358, 0.092224, 0.587811, 0.073116, 0.541977,
              0.36677 , 0.535579, 0.064572, 0.57947 ])
        assert_array_almost_equal(val_error, [0.104392, 0.527721, 0.115536, 0.54828 , 0.09621 , 0.518178,
              0.618695, 0.604389, 0.155411, 0.65563 ])
        assert_array_almost_equal(test_error, [0.180962, 0.534083, 0.126773, 0.659037, 0.150969, 0.639413,
              0.605375, 0.601363, 0.15255 , 0.563358])

        # To print p:
        # print(','.join(np.array2string(y, separator=",", formatter={'float_kind':lambda x: "%.2f" % x})
        # for y in np.array(p.todense())))
        assert_array_almost_equal(p.todense(), [
            [0.36, 0.02, 0.00, -0.10, -0.03], [0.13, -0.01, 0.00, 0.00, 0.12], [-0.05, 0.43, -0.03, -0.10, 0.00],
            [0.00, 0.05, 0.12, 0.08, 0.00], [-0.07, 0.00, 0.45, -0.07, -0.06], [0.02, 0.00, 0.00, 0.02, 0.12],
            [-0.12, 0.09, 0.06, 0.00, 0.16], [0.23, 0.04, -0.12, 0.00, 0.00], [0.00, -0.04, -0.09, -0.01, 0.39],
            [0.00, 0.00, 0.06, -0.06, 0.17]], decimal=2)
