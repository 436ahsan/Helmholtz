"""Multilevel solver (producer of low-residual test functions of the Helmholtz operator."""
import logging
from typing import Tuple

import numpy as np
from numpy.linalg import norm

import helmholtz as hm
from helmholtz.linalg import scaled_norm

_LOGGER = logging.getLogger(__name__)


class Multilevel:
    """The multilevel hierarchy. Contains a sequence of levels."""

    def __init__(self):
        self.level = []

    def __len__(self):
        return len(self.level)

    def relax(self, x: np.array, nu_pre: int, nu_post: int, nu_coarsest: int) -> np.array:
        """
        Executes a relaxation V(nu_pre, nu_post) -cycle on A*x = 0.

        Args:
            x: initial guess. May be a vector of size n or a matrix of size n x m, where A is n x n.
            nu_pre: number of relaxation sweeps at a level before visiting coarser levels.
            nu_post: number of relaxation sweeps at a level after visiting coarser levels.
            nu_coarsest: number of relaxation sweeps to run at the coarsest level.

        Returns:
            x after the cycle.
        """
        # TODO(orenlivne): replace by a general-cycle-index, non-recursive loop.
        return self._relax(0, x, nu_pre, nu_post, nu_coarsest, np.zeros_like(x))

    def _relax(self, level_ind: int, x: np.array, nu_pre: int, nu_post: int, nu_coarsest: int, b: np.ndarray) -> np.array:
        level = self.level[level_ind]
        if level_ind == len(self) - 1:
            # Coarsest level.
            for _ in range(nu_coarsest):
                x = level.relax(x, b)
        else:
            for _ in range(nu_pre):
                x = level.relax(x, b)

            if level_ind < len(self) - 1:
                coarse_level = self.level[level_ind + 1]
                # Full Approximation Scheme (FAS).
                xc_initial = coarse_level.r.dot(x)
                bc = coarse_level.r.dot(b - level.operator(x)) + coarse_level.operator(xc_initial)
                #xc = np.zeros((coarse_level.r.shape[0], x.shape[1]))
                #xc = coarse_level.r.dot(x)
                xc = self._relax(level_ind + 1, xc, nu_pre, nu_post, nu_coarsest, bc)
                x += coarse_level.p.dot(xc - xc_initial)

            for _ in range(nu_post):
                x = level.relax(x, b)

        return x


class Level:
    """A single level in the multilevel hierarchy."""

    def __init__(self, a, r, p, r_csr, p_csr):
        self.a = a
        self.r = r
        self.p = p
        self._r_csr = r_csr
        self._p_csr = p_csr
        self._relaxer = hm.kaczmarz.KaczmarzRelaxer(a)

    @staticmethod
    def create_finest_level(a):
        return Level(a, None, None, None, None)

    @staticmethod
    def create_coarse_level(a, r, p):
        num_aggregates = a.shape[0] // r.shape[1]
        ac = (r_csr.dot(a)).dot(p_csr)
        r_csr = r.tile(num_aggregates)
        p_csr = p.tile(num_aggregates)
        return Level(ac, r, p, r_csr, p_csr)

    def print(self):
        _LOGGER.info("a = \n" + str(self.a.toarray()))
        if self.r is not None:
            _LOGGER.info("r = \n" + str(r))
        if self.p is not None:
            _LOGGER.info("p = \n" + str(p))

    def operator(self, x: np.array) -> np.array:
        """
        Returns the operator action A*x.
        Args:
            x: vector of size n or a matrix of size n x m, where A is n x n.

        Returns:
            A*x.
        """
        return self.a.dot(x)

    def relax(self, x: np.array,  b: np.array) -> np.array:
        """
        Executes a relaxation sweep on A*x = 0 at this level.
        Args:
            x: initial guess. May be a vector of size n or a matrix of size n x m, where A is n x n.
            b: RHS. Same size as x.

        Returns:
            x after relaxation.
        """
        return self._relaxer.step(x, b)

    def restrict(self, x: np.array) -> np.array:
        """
        Returns the restriction action R*x.
        Args:
            x: vector of size n or a matrix of size n x m.

        Returns:
            x^c = R*x.
        """
        return self._r_csr.dot(x)

    def restrict(self, xc: np.array) -> np.array:
        """
        Returns the interpolation action R*x.
        Args:
            xc: vector of size n or a matrix of size nc x m.

        Returns:
            x = P*x^c.
        """
        return self._p_csr.dot(xc)


def relax_test_matrix(operator, method, e: np.ndarray, num_sweeps: int = 30, print_frequency: int = None) -> np.ndarray:
    """
    Creates test functions (functions that approximately satisfy A*x=0) using single level relaxation.

    Args:
        operator: an object that can calculate residuals (action A*x).
        method: iterative method functor (an iteration is a call to this method).
        e: test matrix initial approximation to the test functions.
        num_sweeps: number of sweeps to execute.

    Returns:
        e: relaxed test matrix.
        conv_factor: asymptotic convergence factor (of the last iteration).
    """
    # Print the error and residual norm of the first test function.
    e0 = e[:, 0]
    r_norm = scaled_norm(operator(e0))
    _LOGGER.debug("{:5d} |e| {:.8e} |r| {:.8e}".format(0, scaled_norm(e0), r_norm))

    # Run 'num_sweeps' relaxation sweeps.
    if print_frequency is None:
        print_frequency = num_sweeps // 10
    for i in range(1, num_sweeps + 1):
        r_norm_old = r_norm
        e = method(e)
        e0 = e[:, 0]
        r_norm = scaled_norm(operator(e0))
        if i % print_frequency == 0:
            _LOGGER.debug("{:5d} |e| {:.8e} |r| {:.8e} ({:.5f})".format(
                i, scaled_norm(e0), r_norm, r_norm / r_norm_old))
    # Scale e to unit norm, as we are calculating eigenvectors.
    e /= norm(e)
    return e, r_norm / r_norm_old


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
