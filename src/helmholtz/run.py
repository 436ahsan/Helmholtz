"""Iterative method running and convergence estimation functions."""
import helmholtz as hm
import logging
import numpy as np
from typing import Tuple
from numpy.linalg import norm
from helmholtz.linalg import scaled_norm

_LOGGER = logging.getLogger(__name__)


def run_iterative_method(operator, method, x: np.ndarray, lam, num_sweeps: int = 30, print_frequency: int = None,
                         residual_stop_value: float = 1e-10) -> np.ndarray:
    """
    Creates test functions (functions that approximately satisfy A*x=0) using single level relaxation.

    Args:
        operator: an object that can calculate residuals (action A*x).
        method: iterative method functor (an iteration is a call to this method).
        x: test matrix initial approximation to the test functions.
        num_sweeps: number of sweeps to execute.
        print_frequency: print debugging convergence statements per this number of sweeps.
          None means no printouts.
        residual_stop_value: stop iteration when scaled_norm(operator(x)) < residual_stop_value.

    Returns:
        e: relaxed test matrix.
        conv_factor: asymptotic convergence factor (of the last iteration).
    """
    # Print the error and residual norm of the first test function.
    x0 = x[:, 0]
    r_norm = scaled_norm(operator(x0, lam))
    lam_error = 1
    _LOGGER.debug("{:5d} |r| {:.8e} lam {:.5f}".format(0, r_norm, lam))
    # Run 'num_sweeps' relaxation sweeps.
    if print_frequency is None:
        print_frequency = num_sweeps // 10
    r_norm_history = [None] * (num_sweeps + 1)
    r_norm_history[0] = r_norm
    min_sweeps = 5
    for i in range(1, num_sweeps + 1):
        r_norm_old = r_norm
        lam_old = lam
        lam_error_old = lam_error
        x, lam = method(x, lam)
        x0 = x[:, 0]
        r_norm = scaled_norm(operator(x0, lam))
        lam_error = np.abs(lam - lam_old)
        if i % print_frequency == 0:
            _LOGGER.debug("{:5d} |r| {:.8e} ({:.5f}) lam {:.5f} ({:.5f})".format(
                i, r_norm, r_norm / max(1e-30, r_norm_old), lam, lam_error / max(1e-30, lam_error_old)))
        r_norm_history[i] = r_norm
        if i >= min_sweeps and r_norm < residual_stop_value:
            r_norm_history = r_norm_history[:i + 1]
            break
    # return x, r_norm / r_norm_old
    # Average convergence factor over the last 5 steps. Exclude first cycle.
    last_steps = min(5, len(r_norm_history) - 2)
    return x, lam, (r_norm_history[-1] / r_norm_history[-last_steps - 1]) ** (1 / last_steps)


def random_test_matrix(window_shape: Tuple[int], num_examples: int = None) -> np.ndarray:
    """
    Creates the initial test functions as random[-1, 1].

    Args:
        window_shape: domain size (#gridpoints in each dimension).
        num_examples: number of test functions to generate.

    Returns:
        e: window_size x num_examples random test matrix.
    """
    if num_examples is None:
        # By default, use more test functions than gridpoints so we have a sufficiently large test function sample.
        num_examples = 4 * np.prod(window_shape)

    # Start from random[-1,1] guess for each function.
    e = 2 * np.random.random(window_shape + (num_examples,)) - 1
    e /= norm(e)
    return e
