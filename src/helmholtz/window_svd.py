"""Calculates the SVD of a test matrix on a window, which gives us the coarse variables definition (R)."""
import logging
from typing import Tuple

import numpy as np
from numpy.linalg import svd

import helmholtz as hm

logger = logging.getLogger(__name__)


def get_window_svd(a,
                   window_shape: Tuple[int],
                   num_examples: int = None,
                   num_sweeps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the SVD of a test matrix e produced by single-level relaxation.

    Args:
        a: Helmholtz operator.
        window_shape: domain size (#gridpoints in each dimension).
        num_examples: number of test functions to generate. If None, uses 4 * np.prod(window_shape).
        num_sweeps: number of sweeps to execute.

    Returns:
        s: singular values in descending order.
        vh: corresponding V^T matrix of right singular vectors. The kth singular vector is the row vh[k].
    """
    if num_examples is None:
        # By default, use more test functions than gridpoints so we have a sufficiently large test function sample.
        num_examples = 4 * np.prod(window_shape)
    x = hm.run.random_test_matrix(window_shape, num_examples=num_examples)
    level = hm.multilevel.Level.create_finest_level(a)
    b = np.zeros_like(x)
    lam = 0
    x, lam, _ = hm.run.run_iterative_method(level.operator, lambda x, lam: (level.relax(x, b, lam), lam),
                                            x, lam, num_sweeps=num_sweeps)
    # Calculate the SVD.
    e_matrix = np.reshape(x, [np.prod(window_shape), num_examples])
    _, s, vh = svd(e_matrix.transpose())
    return s / s[0], vh
