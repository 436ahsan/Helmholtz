"""Calculates the SVD of a test matrix on a window with periodic boundary conditions."""
import logging
import numpy as np
from numpy.linalg import svd, norm
from typing import Tuple


def scaled_norm(e: np.ndarray) -> float:
    """
    Returns the scaled L2 norm of a test function e:

     [ sum(e[i1,...,id] ** 2 for all (i1,...,id)) / np.prod(e.shape) ] ** 0.5

    Args:
        e: test function, where e.shape[d] = #gridpoints in dimension d.

    Returns:
        The scaled L2 norm of e.
    """
    return norm(e) / np.prod(e.shape) ** 0.5


def get_window_svd(operator, relaxer, window_shape: tuple, num_sweeps: int = 30, num_examples: int = None) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Returns the SVD of a test matrix e.

    Args:
        operator: A functor that calculates the operator L(e).
        relaxer: A functor that executes relaxation in place on e.
        window_shape: domain size (#gridpoints in each dimension).
        num_sweeps: number of sweeps to execute. If None, uses 4 * np.prod(window_shape).
        num_examples: number of test functions to generate.

    Returns:
        s: singular values in descending order.
        vh: corresponding V^T matrix of right singular vectors. The kth singular vector is the row vh[k].
    """
    logger = logging.getLogger("get_window_svd")
    if num_examples is None:
        # By default, use more test functions than gridpoints so we have a sufficiently large test function sample.
        num_examples = 4 * np.prod(window_shape)

    # Start from random[-1,1] guess for each function.
    e = 2 * np.random.random(window_shape + (num_examples,)) - 1
    # Print the error and residual norm of the first test function.
    # A poor way of getting the last "column" of the tensor e.
    e0 = e.reshape(-1, e.shape[-1])[:, 0].reshape(e.shape[:-1])
    logger.debug("{:5d} |e| {:.8e} |r| {:.8e}".format(0, scaled_norm(e0), scaled_norm(operator(e0))))

    # Run 'num_sweeps' relaxation sweeps.
    for i in range(1, num_sweeps + 1):
        relaxer(e)
        if i % (num_sweeps // 10) == 0:
            # A poor way of getting the last "column" of the tensor e.
            e0 = e.reshape(-1, e.shape[-1])[:, 0].reshape(e.shape[:-1])
            logger.debug("{:5d} |e| {:.8e} |r| {:.8e}".format(i, scaled_norm(e0), scaled_norm(operator(e0))))
        # Scale e to unit norm to avoid underflow, as we are calculating eigenvectors.
        e /= norm(e)

    # Calculate the SVD.
    e_matrix = np.reshape(e, [np.prod(window_shape), num_examples])
    _, s, vh = svd(e_matrix.transpose())
    return s / s[0], vh