import logging
import helmholtz as hm
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy import optimize


_LOGGER = logging.getLogger(__name__)


def optimal_rotation_angle(xc):
    # Sample concecutive pairs of coarse variables. Because of the ordering of the coarse
    # variables in aggregates over the domain as (0, 1), (2, 3), (4, 5), ..., this means a
    # super-aggregate size = 4. X then corresponds to (0, 1) and Y to (2, 3).
    # Note that window offsets must be even (we don't want to include windows like (1, 2)).
    coarse_aggregate_size = 4

    xc_aggregate_t = np.concatenate(
        tuple(hm.linalg.get_window(xc, offset, coarse_aggregate_size)
              for offset in range(0, max((4 * coarse_aggregate_size) // xc.shape[1], 1), 2)), axis=1).transpose()

    X, Y = xc_aggregate_t[:2], xc_aggregate_t[2:]

    f = lambda t: get_local_rotation_min_function(X, Y, t, scale=False, p=2)
    tmin = optimize.minimize_scalar(f, bounds=(-np.pi, np.pi), method='brent').x % (2 * np.pi)
    # tmin = optimize.shgo(f, [(-np.pi, np.pi)], sampling_method='sobol').x[0]
    f0 = f(0)
    fmin = f(tmin)

    print("{:<+3f} {:2e} {:2e} ({:6.2f})".format(tmin / np.pi, f0, fmin, f0 / fmin))
    return f, tmin


def calculate_local_rotation_angles(n, aggregate_size, nc, xc):
    """Returns the rotation angle of X_{i+1} with respect to X_i, where X_i is the pair of coarse variables
    of aggregate i."""
    # Number of aggregates.
    N = n // aggregate_size
    num_coarse = nc * N
    p = 2

    phi = []
    _LOGGER.info("{:<3s} {:<9s} {:<12s} {:<12s} {:<8s} {:<13s} {:<13s}".format(
        "Agg", "t/pi", "f0", "fmin", "factor", "Dist Before", "Dist After"))
    for j, i in enumerate(range(0, num_coarse, nc)):
        f = lambda t: get_local_rotation_min_function(
            xc[np.arange(i + nc, i + 2*nc) % num_coarse], xc[i:i + nc], t, scale=False, p=p)
        result = optimize.minimize_scalar(f, bounds=(-np.pi, np.pi), method='brent')
        tmin = result.x % (2 * np.pi)
        #tmin = optimize.shgo(f, [(-np.pi, np.pi)], sampling_method='sobol').x[0]
        f0 = f(0)
        fmin = f(tmin)
        norm_before = norm(xc[i:i + nc] - xc[np.arange(i + nc, i + 2*nc) % num_coarse], axis=1) #** 2
        u = rotation(tmin)
        norm_after = norm(u.dot(xc[i:i + nc]) - xc[np.arange(i + nc, i + 2*nc) % num_coarse], axis=1) #** 2
        _LOGGER.info("{:<3d} {:<+3f} {:2e} {:2e} ({:6.2f}) {} {}".format(
            i, tmin / np.pi, f0, fmin, f0 / fmin, norm_before, norm_after))
        phi.append(tmin)
    phi = np.array(phi)
    return phi


def calculate_local_repetitive_rotation_angle(num_aggregates, nc, xc):
    # Number of aggregates.
    num_coarse = nc * num_aggregates
    p = 2

    phi = []
    _LOGGER.info("{:<3s} {:<9s} {:<12s} {:<12s} {:<8s} {:<13s} {:<13s}".format(
        "Agg", "t/pi", "f0", "fmin", "factor", "Dist Before", "Dist After"))

    f = lambda t: sum(get_local_rotation_min_function(
            xc[i:i + nc], xc[np.arange(i + nc, i + 2*nc) % num_coarse], t, scale=False, p=p)
            for i in range(0, num_coarse, nc))
    result = optimize.minimize_scalar(f, bounds=(-np.pi, np.pi), method='brent')
    tmin = result.x % (2 * np.pi)

    for j, i in enumerate(range(0, num_coarse, nc)):
        f = lambda t: get_local_rotation_min_function(
            xc[i:i + nc], xc[np.arange(i + nc, i + 2 * nc) % num_coarse], t, scale=False, p=p)
        result = optimize.minimize_scalar(f, bounds=(-np.pi, np.pi), method='brent')
        tmin_i = result.x % (2 * np.pi)
        #tmin_i = optimize.shgo(f, [(-np.pi, np.pi)], sampling_method='sobol').x[0]
        f0 = f(0)
        fmin = f(tmin)
        norm_before = norm(xc[i:i + nc] - xc[np.arange(i + nc, i + 2*nc) % num_coarse], axis=1) #** 2
        u = rotation(tmin)
        norm_after = norm(u.dot(xc[i:i + nc]) - xc[np.arange(i + nc, i + 2*nc) % num_coarse], axis=1) #** 2

        fmin_i = f(tmin_i)
        u = rotation(tmin_i)
        norm_after_i = norm(u.dot(xc[i:i + nc]) - xc[np.arange(i + nc, i + 2*nc) % num_coarse], axis=1) #** 2

        _LOGGER.info("{:<3d} {:<+3f} {:6.2e} {:6.2e} ({:6.2f}) {} {} {:<+3f} {:6.2e} ({:6.2f}) {} {}".format(
            i, tmin / np.pi, f0,
            fmin, f0 / fmin, norm_before, norm_after,
            tmin_i / np.pi, fmin_i, f0 / fmin_i, norm_before, norm_after))
    return tmin


def rotation(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s], [-s, c]])


def get_local_rotation_min_function(X, Y, t, scale=False, p=2):
    if scale:
        scale_x = (0.5 * (norm(X[:, 0]) + norm(X[:, 1]))) ** 0.5
        scale0 = norm(Y[:, 0]) ** 2 * scale_x
        scale1 = norm(Y[:, 1]) ** 2 * scale_x
    else:
        scale0 = scale1 = 1
    return \
        norm( np.cos(t) * X[0] + np.sin(t) * X[1] - Y[0]) ** p / scale0 + \
        norm(-np.sin(t) * X[0] + np.cos(t) * X[1] - Y[1]) ** p / scale1


def get_local_rotation_angle(X, Y):
    return optimize.minimize_scalar(lambda t: get_local_rotation_min_function(X, Y, t),
                                    bounds=(-np.pi, np.pi), method='brent').x


def plot_min_functions(n, aggregate_size, nc, xc, ax):
    N = n // aggregate_size
    num_coarse = nc * N
    p = 2
    t = np.linspace(0, 2 * np.pi, 100)
    for j, i in enumerate(range(0, num_coarse, nc)):
        f = lambda t: get_local_rotation_min_function(
            xc[i:i + nc], xc[np.arange(i + nc, i + 2*nc) % num_coarse], t, scale=False, p=p)
        if j < 10:
            ax.plot(t / np.pi, np.array([f(theta) for theta in t]), label="Agg {}".format(j))
    ax.grid(True);
    ax.set_xlabel(r"$\theta / \pi$")
    #ax.legend();
