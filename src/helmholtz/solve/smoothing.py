"""Direct numerical predictor mimicking the smoothing factor."""
import helmholtz as hm
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy import optimize


_LOGGER = logging.getLogger(__name__)


def shrinkage_factor(operator, method, domain_shape: np.ndarray, num_examples: int = 5,
                     slow_conv_factor: float = 0.95, print_frequency: int = None, max_sweeps: int = 100,
                     leeway_factor: float = 1.2) -> float:
    """
    Returns the shrinkage factor of an iterative method, the residual-to-error ratio (RER) reduction in the first
    num_sweeps steps for A*x = 0, starting from an initial guess.

    Args:
        operator: an object that can calculate residuals (action A*x).
        method: solve method functor (an iteration is a call to this method).
        method: method object to run.
        domain_shape: shape of input vector to relaxation.
        num_examples: # experiments (random starts) to average over.
        slow_conv_factor: stop when convergence factor exceeds this number.
        print_frequency: print debugging convergence statements per this number of sweeps.
        max_sweeps: maximum number of iterations to run.
        leeway_factor: efficiency inflation factor past the point of diminishing returns to use in estimating where
            slowness starts.

    Returns:
        mu: shrinkage factor.
        index: Point of Diminishing Returns (PODR) index into the residual_history array.
        residual_history: residual norm run history array.
        conv_history: residual norm convergence factor run history array.
        conv_factor: asymptotic convergence factor estimate. This is only good for detecting a strong divergence or
            convergence, and not meant to be quantitatively accurate for slow, converging iterations.
    """
#    assert num_examples > 1
    x = hm.solve.run.random_test_matrix(domain_shape, num_examples=num_examples)
    x_norm = norm(x, axis=0)
    r_norm = norm(operator(x), axis=0)
    rer = r_norm / x_norm
    if print_frequency is not None:
        _LOGGER.info("Iter     |r|                 |x|         RER")
        _LOGGER.info("{:<5d} {:.3e}            {:.3e}    {:.3f}".format(
            0, np.mean(r_norm), np.mean(x_norm), np.mean(rer)))
    b = np.zeros_like(x)
    rer_conv_factor = 0
    i = 0
    residual_history = [r_norm]
    # reduction = [np.ones_like(r_norm)]
    # efficiency = list(reduction[0] ** (1 / 1e-2))
    while rer_conv_factor < slow_conv_factor and i < max_sweeps:
        i += 1
        r_norm_old = r_norm
        rer_old = rer
        x = method(x, b)
        r_norm = norm(operator(x), axis=0)
        x_norm = norm(x, axis=0)
        rer = r_norm / x_norm
        rer_conv_factor = np.mean(rer / np.clip(rer_old, 1e-30, None))
        if print_frequency is not None and i % print_frequency == 0:
            _LOGGER.info("{:<5d} {:.3e} ({:.3f})    {:.3e}    {:.3f} ({:.3f})".format(
                i, np.mean(r_norm), np.mean(r_norm / np.clip(r_norm_old, 1e-30, None)), np.mean(x_norm),
                np.mean(rer), rer_conv_factor))
        residual_history.append(r_norm)
        # red = np.mean(r_norm / history[0])
        # reduction.append(red)
        # efficiency.append(red ** (1 / i))
    residual_history = np.array(residual_history)
    # reduction = np.array(reduction)
    # efficiency = np.array(efficiency)

    # Find point of diminishing returns (PODR). Allow a leeway of 'leeway_factor' from the maximum efficiency point.
    reduction = np.mean(residual_history / residual_history[0], axis=1)
    efficiency = reduction ** (1 / np.clip(np.arange(residual_history.shape[0]), 1e-2, None))
    index = max(np.where(efficiency < leeway_factor * min(efficiency))[0])
    # factor = residual reduction per sweep over the first 'index' sweeps.
    factor = efficiency[index]

    # Estimate the asymptotic convergence factor at twice the PODR.
    # Residual convergence factor history.
    conv_history = np.mean(np.exp(np.diff(np.log(residual_history), axis=0)), axis=1)
    conv_factor = conv_history[min(2 * index, len(conv_history) - 1)]
    return factor, index, residual_history, conv_history, conv_factor


def _conv_model(x, x0, y0, c, p):
    return np.piecewise(x, [x < x0],
                        [lambda x: y0,
                         lambda x: c - (c - y0)*(x / x0) ** p
                        ])


def _fit_conv_model(conv):
    x = np.arange(1, len(conv) + 1)
    y = conv
    p0 = [np.argmin(y), np.min(y), x[-1], -1]
    p, _ = optimize.curve_fit(_conv_model, x, y, p0=p0, maxfev=5000)
    return p


def plot_fitted_conv_model(p, conv, ax, title: str = "Relax"):
    x = np.arange(1, len(conv) + 1)
    ax.plot(x, conv, "o")
    xd = np.linspace(1, len(conv) + 1, 100)
    ax.plot(xd, _conv_model(xd, *p), label=r"{} $\mu_0 = {:.2f}, n_0 = {:.2f}, p = {:.1f}$".format(
        title, p[1], p[0], p[3]))
    ax.set_ylabel("RER Reduction")
    ax.set_xlabel("Sweep #")
    ax.grid(True)


def plot_diminishing_returns_point(factor, num_sweeps, conv, ax, title: str = "Relax", color: str = "b"):
    x = np.arange(1, len(conv) + 1)
    ax.plot(x, conv, "o", color=color, label=r"{} $\mu = {:.2f}, i = {}$".format(title, factor, num_sweeps))
    ax.scatter([num_sweeps], [conv[num_sweeps - 1]], 120, facecolors='none', edgecolors=color)
    ax.set_ylabel("Residual Reduction Factor")
    ax.set_xlabel("Sweep #")
    ax.grid(True)


def check_relax_cycle_shrinkage(multilevel, max_sweeps: int = 20, num_levels: int = None,
                                nu_pre: int = 2, nu_post: int = 2, nu_coarsest: int = 4):
    """Checks the two-level relaxation cycle shrinkage vs. relaxation."""
    level = multilevel.level[0]
    a = level.a
    def relax_cycle(x):
        return hm.solve.relax_cycle.relax_cycle(multilevel, 1.0, nu_pre, nu_post, nu_coarsest,
                                                num_levels=num_levels).run(x)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    num_examples = 5
    b = np.zeros((a.shape[0], num_examples))

    operator = lambda x: a.dot(x)
    relax = lambda x: level.relax(x, b)

    relax_b = lambda x, b: level.relax(x, b)
    relax_cycle_b = lambda x, b: relax_cycle(x)

    for title, method, method_b, work, color in zip(
        ("Kaczmarz", "Mini-cycle"), (relax, relax_cycle), (relax_b, relax_cycle_b), (1, 6), ("blue", "red")):
        #print(title)
        y, conv_factor = hm.solve.run.run_iterative_method(
            level.operator, method, hm.solve.run.random_test_matrix((level.size, ), num_examples=1),
            30,  print_frequency=None)

        factor, num_sweeps, r, conv = hm.solve.smoothing.shrinkage_factor(
            operator, method_b, (a.shape[0], ), print_frequency=1, max_sweeps=max_sweeps, slow_conv_factor=1.1, leeway_sweeps=3)
        hm.solve.smoothing.plot_diminishing_returns_point(factor, num_sweeps, conv, ax, title=title, color=color)
        print("{:<10s} RER at point of diminishing returns {:.2f} num_sweeps {:>2d} work {:>2d} Residual-per-sweep {:.2f}".format(
            title, np.mean(r[num_sweeps]), num_sweeps, work, np.mean(r[num_sweeps] / r[0]) ** (1/(num_sweeps * work))))

    ax.legend();
    # y, conv_factor = hm.solve.run.run_iterative_method(
    #     level.operator, relax_cycle, hm.solve.run.random_test_matrix((level.size, ), num_examples=1),
    #     10,  print_frequency=1)
    # y_all[num_levels] = y
    # print("Conv Factor {:.5f}".format(conv_factor))