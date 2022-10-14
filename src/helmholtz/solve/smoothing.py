"""Direct numerical predictor mimicking the smoothing factor."""
import helmholtz as hm
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy import optimize


_LOGGER = logging.getLogger(__name__)


def shrinkage_factor(operator, method, domain_shape: np.ndarray, num_examples: int = 5,
                     slow_conv_factor: float = 1.3, print_frequency: int = None, max_sweeps: int = 100,
                     leeway_factor: float = 1.2, min_residual_reduction: float = 0.2,
                     output: str = "stats", x0: np.ndarray = None) -> float:
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
        min_residual_reduction: minimum residual norm reduction required for PODR.
        output: if "stats", outputs the main stats (5 fields of the return values documented below). If "history",
            also outputs the x and residual (A*x) history.
        x0: optional initial guess for the method run on A*x=0.

    Returns:
        mu: shrinkage factor.
        index: Point of Diminishing Returns (PODR) index into the residual_history array.
        residual_history: residual norm run history array.
        conv_history: residual norm convergence factor run history array.
        conv_factor: asymptotic convergence factor estimate. This is only good for detecting a strong divergence or
            convergence, and not meant to be quantitatively accurate for slow, converging iterations.
        x_history: list of x iterates (x_history[0] is the initial guess, etc.).
            Outputted only if output="history".
        r_history: list of residual iterates (r_history[0] is the initial residual, etc.).
            Outputted only if output="history".
    """
#    assert num_examples > 1
    if x0 is None:
        x = hm.solve.run.random_test_matrix(domain_shape, num_examples=num_examples)
    else:
        x = x0
    r = operator(x)
    x_history = []
    r_history = []
    if output == "history":
        x_history.append(x)
        r_history.append(r)
    x_norm = norm(x, axis=0)
    r_norm = norm(r, axis=0)
    rer = r_norm / x_norm
    if print_frequency is not None:
        _LOGGER.info("Iter     |r|                         |x|         RER")
        _LOGGER.info("{:<5d} {:.3e}                    {:.3e}    {:.3f}".format(
            0, np.mean(r_norm), np.mean(x_norm), np.mean(rer)))
    b = np.zeros_like(x)
    rer_conv_factor = 0
    residual_conv_factor = 0
    i = 0
    residual_history = [r_norm]
    rer_history = [rer]
    # reduction = [np.ones_like(r_norm)]
    # efficiency = list(reduction[0] ** (1 / 1e-2))
    while residual_conv_factor < slow_conv_factor and i < max_sweeps:
        i += 1
        r_norm_old = r_norm
        rer_old = rer
        x = method(x, b)
        r = operator(x)
        if output == "history":
            x_history.append(x)
            r_history.append(r)
        x_norm = norm(x, axis=0)
        r_norm = norm(r, axis=0)
        rer = r_norm / x_norm
        rer_conv_factor = np.mean(rer / np.clip(rer_old, 1e-30, None))
        residual_conv_factor = np.mean(r_norm / np.clip(r_norm_old, 1e-30, None))
        if print_frequency is not None and i % print_frequency == 0:
            _LOGGER.info("{:<5d} {:.3e} ({:.3f}) [{:.3f}]    {:.3e}    {:.3f} ({:.3f})".format(
                i, np.mean(r_norm), residual_conv_factor,
                np.mean(r_norm / residual_history[0]) ** (1 / i),
                np.mean(x_norm),
                np.mean(rer), rer_conv_factor))
        residual_history.append(r_norm)
        rer_history.append(rer)
        # rer = np.mean(r_norm / history[0])
        # reduction.append(rer)
        # efficiency.append(rer ** (1 / i))
    residual_history = np.array(residual_history)
    # reduction = np.array(reduction)
    # efficiency = np.array(efficiency)

    # Find point of diminishing returns (PODR). Allow a leeway of 'leeway_factor' from the maximum efficiency point.
    reduction = np.mean(residual_history / residual_history[0], axis=1)
    efficiency = reduction ** (1 / np.clip(np.arange(residual_history.shape[0]), 1e-2, None))
    sufficient_reduction_index = min(np.where(reduction < min_residual_reduction)[0])
    index = max(np.where(efficiency < leeway_factor * min(efficiency))[0])
    index = max(index, sufficient_reduction_index)
    # factor = residual reduction per sweep over the first 'index' sweeps.
    factor = efficiency[index]

    # Estimate the asymptotic convergence factor at twice the PODR.
    # Residual convergence factor history.
    conv_history = np.mean(np.exp(np.diff(np.log(residual_history), axis=0)), axis=1)
    conv_factor = conv_history[min(max(10, 2 * index), len(conv_history) - 1)]
    result = (factor, index, residual_history, conv_history, rer_history, conv_factor)
    if output == "history":
        result = result + (x_history, r_history)
    return result


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
                                nu_pre: int = 2, nu_post: int = 2, nu_coarsest: int = 4,
                                slow_conv_factor: float = 0.95, leeway_factor: float = 1.2,
                                num_examples: int = 5, x0: np.ndarray = None,
                                print_frequency: int = 1, plot: bool = True):
    """Checks the two-level relaxation cycle shrinkage vs. relaxation, unless num_levels=1, in which case
    we only run relaxation."""
    level = multilevel[0]
    a = level.a
    n = a.shape[0]
    b = np.zeros((n, num_examples))
    operator = lambda x: a.dot(x)
    relax = lambda x: level.relax(x, b)
    relax_b = lambda x, b: level.relax(x, b)
    method_list = [("relax", relax, relax_b, 1, "blue")]

    if num_levels >= 2:
        def relax_cycle(x):
            return hm.solve.relax_cycle.relax_cycle(multilevel, 1.0, nu_pre, nu_post, nu_coarsest,
                                                    num_levels=num_levels).run(x)
        # This is two-level work.
        # TODO(orenlivne): generalize to multilevel work.
        r = multilevel[1]._r
        work = nu_pre + nu_post + (r.shape[0] / r.shape[1]) * nu_coarsest
        relax_cycle_b = lambda x, b: relax_cycle(x)
        method_list.append(("{}-level MiniCycle".format(num_levels), relax_cycle, relax_cycle_b, work, "red"))

    if plot:
        fig, axs = plt.subplots(len(method_list), 5, figsize=(18, 4 * len(method_list)))

    method_info = {}
    for row, (title, method, method_b, work, color) in enumerate(method_list):
        if plot:
            ax_row = axs[row] if len(method_list) > 1 else axs
            _LOGGER.info(title)
        info = hm.solve.smoothing.shrinkage_factor(
            operator, method_b, (n,), print_frequency=print_frequency, max_sweeps=max_sweeps,
            slow_conv_factor=slow_conv_factor, leeway_factor=leeway_factor, output="history", x0=x0)
        method_info[title] = info

        factor, num_sweeps, residual, conv, rer, relax_conv_factor, x_history, r_history = info
        if plot:
            _LOGGER.info(
                "Relax conv {:.2f} shrinkage {:.2f} PODR RER {:.2f} after {} sweeps. Work {:.1f} eff {:.2f}".format(
                    relax_conv_factor, factor, np.mean(rer[num_sweeps]), num_sweeps, work,
                    np.mean(residual[num_sweeps] / residual[0]) ** (1 / (num_sweeps * work))))
            hm.solve.smoothing.plot_diminishing_returns_point(factor, num_sweeps, conv, ax_row[0], title=title, color=color)

            ax = ax_row[1]
            ax.plot(x_history[0])
            ax.set_title("Initial Error")

            ax = ax_row[2]
            ax.plot(x_history[num_sweeps])
            ax.set_title("Error after {} sweeps".format(num_sweeps))

            ax = ax_row[3]
            ax.plot(r_history[0])
            ax.set_title("Initial Residual")

            ax = ax_row[4]
            ax.plot(r_history[num_sweeps])
            ax.set_title("Residual after {} sweeps".format(num_sweeps))

            ax.legend()
    return method_info