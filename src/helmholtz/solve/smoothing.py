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
                     leeway_sweeps: int = 3) -> float:
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
        leeway_sweeps: number of sweeps to allow past the point of diminishing returns in estimating where slowness
            starts.

    Returns:
        e: relaxed test matrix.
        conv_factor: asymptotic convergence factor (of the last iteration) of the RER.
    """
#    assert num_examples > 1
    x = hm.solve.run.random_test_matrix(domain_shape, num_examples=num_examples)
    r_norm = norm(operator(x), axis=0)
    rer = r_norm / norm(x, axis=0)
    if print_frequency is not None:
        _LOGGER.info("{:5d} |r| {:.8e} RER {:.5f}".format(0, np.mean(r_norm), np.mean(rer)))
    b = np.zeros_like(x)
    conv_factor = 0
    i = 0
    history = [r_norm]
    while conv_factor < slow_conv_factor and i < max_sweeps:
        i += 1
        r_norm_old = r_norm
        rer_old = rer
        x = method(x, b)
        r_norm = norm(operator(x), axis=0)
        rer = r_norm / norm(x, axis=0)
        conv_factor = np.mean(rer / np.clip(rer_old, 1e-30, None))
        if print_frequency is not None and i % print_frequency == 0:
            _LOGGER.info("{:5d} |r| {:.8e} ({:.5f}) RER {:.5f} ({:.5f}) {:.5f}".format(
                i, np.mean(r_norm), np.mean(r_norm / np.clip(r_norm_old, 1e-30, None)),
                np.mean(rer), conv_factor, np.mean(norm(x, axis=0))))
        history.append(r_norm)
    history = np.array(history)

    # Find point of diminishing returns. Allow for leeway of 10% from the maximum efficiency point.
    reduction = np.mean(history / history[0], axis=1)
    efficiency = reduction ** (1 / np.clip(np.arange(history.shape[0]), 1e-2, None))
    index = min(np.argmin(efficiency) + leeway_sweeps, len(efficiency) - 1)
    factor = efficiency[index]
    # Residual convergence factor.
    conv = np.mean(np.exp(np.diff(np.log(history), axis=0)), axis=1)
    return factor, index, history, conv


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
    ax.set_ylabel("Residual Reduction")
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