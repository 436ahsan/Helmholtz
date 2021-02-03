"""Multilevel solver (producer of low-residual test functions of the Helmholtz operator."""
import logging
from typing import Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
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

    def relax_cycle(self, x: np.array, nu_pre: int, nu_post: int, nu_coarsest: int, debug: bool = False,
                    update_lam: str = "finest", finest_level_ind: int = 0) -> np.array:
        """
        Executes a relaxation V(nu_pre, nu_post) -cycle on A*x = 0.

        Args:
            x: initial guess. May be a vector of size n or a matrix of size n x m, where A is n x n.
            nu_pre: number of relaxation sweeps at a level before visiting coarser levels.
            nu_post: number of relaxation sweeps at a level after visiting coarser levels.
            nu_coarsest: number of relaxation sweeps to run at the coarsest level.
            debug: print logging debugging printouts or not.

        Returns:
            x after the cycle.
        """
        # TODO(orenlivne): replace by a general-cycle-index, non-recursive loop. Move relaxation, transfers logic
        # to a processor separate from the cycle.
        if debug:
            _LOGGER.debug("-" * 80)
        sigma = np.ones(x.shape[1], )
        b = np.zeros_like(x)
        x = self._relax_cycle(finest_level_ind, x, sigma, nu_pre, nu_post, nu_coarsest, b, debug, update_lam)
        if update_lam == "finest":
            x = self._update_global_constraints(x, sigma, b, self.level[finest_level_ind])
        return x

    def _update_global_constraints(self, x, sigma, b, level):
        """
        Updates lambda + normalize at level 'level'.
        Args:
            x:

        Returns:
            Updated x. Global lambda also updated.
        """
        eta = level.normalization(x)
        # TODO(orenlivne): vectorize the following expressions.
        for i in range(x.shape[1]):
            x[:, i] *= (sigma[i] / eta[i]) ** 0.5
        level.global_params.lam = np.mean([level.rq(x[:, i], b[:, i]) for i in range(x.shape[1])])
        return x

    def _relax_cycle(self, level_ind: int, x: np.array, sigma: float,
                     nu_pre: int, nu_post: int, nu_coarsest: int, b: np.ndarray, debug: bool, update_lam: str) -> np.array:
        level = self.level[level_ind]
        def print_state(title):
            if debug:
                _LOGGER.debug("{:2d} {:<15} {:.4e} {:.4e}".format(
                    level_ind, title, scaled_norm(b[:, 0] - level.operator(x[:, 0])),
                    np.abs(sigma - level.normalization(x))[0]))

        print_state("initial")
        if level_ind == len(self) - 1:
            # Coarsest level.
            for _ in range(nu_coarsest):
                for _ in range(1 if len(self) == 1 else 5):
                    x = level.relax(x, b)
                if update_lam == "coarsest":
                    # Update lambda + normalize only once per several relaxations if multilevel and updating lambda
                    # at the coarsest level.
                    x = self._update_global_constraints(x, sigma, b, level)
            print_state("coarsest ({})".format(nu_coarsest))
        else:
            print_state("relax {}".format(nu_pre))
            if level_ind < len(self) - 1:
                coarse_level = self.level[level_ind + 1]
                # Full Approximation Scheme (FAS).
                xc_initial = coarse_level.restrict(x)
                bc = coarse_level.restrict(b - level.operator(x)) + coarse_level.operator(xc_initial)
                sigma_c = sigma - level.normalization(x) + coarse_level.normalization(xc_initial)
                xc = self._relax_cycle(level_ind + 1, xc_initial, sigma_c, nu_pre, nu_post, nu_coarsest, bc, debug, update_lam)
                x += coarse_level.interpolate(xc - xc_initial)
                print_state("correction {}".format(nu_pre))

            for _ in range(nu_post):
                x = level.relax(x, b)
            print_state("relax {}".format(nu_post))
        return x


class GlobalParams:
    """Parameters shared by all levels of the multilevel hierarchy."""

    def __init__(self):
        # Eigenvalue.
        self.lam = 0


class Level:
    """A single level in the multilevel hierarchy."""

    def __init__(self, a, b, global_params, r, p, r_csr, p_csr):
        self.a = a
        self.b = b
        self.r = r
        self.p = p
        self.global_params = global_params
        self._r_csr = r_csr
        self._p_csr = p_csr
        self._relaxer = hm.relax.KaczmarzRelaxer(a, b)

    @staticmethod
    def create_finest_level(a):
        return Level(a, scipy.sparse.eye(a.shape[0]), GlobalParams(), None, None, None, None)

    @staticmethod
    def create_coarse_level(a, b, global_params, r, p):
        num_aggregates = a.shape[0] // r.asarray().shape[1]
        r_csr = r.tile(num_aggregates)
        p_csr = p.tile(num_aggregates)
        # Form Galerkin coarse-level operator.
        ac = (r_csr.dot(a)).dot(p_csr)
        bc = (r_csr.dot(b)).dot(p_csr)
        return Level(ac, bc, global_params, r, p, r_csr, p_csr)

    def print(self):
        _LOGGER.info("a = \n" + str(self.a.toarray()))
        if self.r is not None:
            _LOGGER.info("r = \n" + str(self.r))
        if self.p is not None:
            _LOGGER.info("p = \n" + str(self.p))

    def stiffness_operator(self, x: np.array) -> np.array:
        """
        Returns the operator action A*x.
        Args:
            x: vector of size n or a matrix of size n x m, where A is n x n.

        Returns:
            A*x.
        """
        return self.a.dot(x)

    def mass_operator(self, x: np.array) -> np.array:
        """
        Returns the operator action B*x.
        Args:
            x: vector of size n or a matrix of size n x m, where B is n x n.

        Returns:
            B*x.
        """
        return self.b.dot(x)

    def operator(self, x: np.array) -> np.array:
        """
        Returns the operator action (A-lam*B)*x. lam is the global eigenvalue value in level.global_params.
        Args:
            x: vector of size n or a matrix of size n x m, where A, B are n x n.

        Returns:
            (A-B*lam)*x.
        """
        return self.a.dot(x) - self.global_params.lam * self.b.dot(x)

    def normalization(self, x: np.array) -> np.array:
        """
        Returns the eigen-normalization functional (Bx, x).
        Args:
            x: vector of size n or a matrix of size n x m, where A, B are n x n.

        Returns:
           (Bx, x) for each column of x.
        """
        return np.array([(self.b.dot(x[:, i])).dot(x[:, i]) for i in range(x.shape[1])])

    def rq(self, x: np.array, b: np.array = None) -> np.array:
        """
        Returns the Rayleigh Quotient of x.
        Args:
            x: vector of size n or a matrix of size n x m, where A, B are n x n.
            b: RHS vector, for FAS coarse problems.

        Returns:
           (Ax, x) / (Bx, x) or ((Ax - b), x) / (Bx, x) if b is not None.
        """
        if b is None:
            return (self.a.dot(x)).dot(x) / (self.b.dot(x)).dot(x)
        else:
            return (self.a.dot(x) - b).dot(x) / (self.b.dot(x)).dot(x)

    def relax(self, x: np.array, b: np.array) -> np.array:
        """
        Executes a relaxation sweep on A*x = 0 at this level.
        Args:
            x: initial guess. May be a vector of size n or a matrix of size n x m, where A is n x n.
            b: RHS. Same size as x.

        Returns:
            x after relaxation.
        """
        return self._relaxer.step(x, b, lam=self.global_params.lam)

    def restrict(self, x: np.array) -> np.array:
        """
        Returns the restriction action R*x.
        Args:
            x: vector of size n or a matrix of size n x m.

        Returns:
            x^c = R*x.
        """
        return self._r_csr.dot(x)

    def interpolate(self, xc: np.array) -> np.array:
        """
        Returns the interpolation action R*x.
        Args:
            xc: vector of size n or a matrix of size nc x m.

        Returns:
            x = P*x^c.
        """
        return self._p_csr.dot(xc)


def relax_test_matrix(operator, rq, method, x: np.ndarray, num_sweeps: int = 30, print_frequency: int = None,
                      residual_stop_value: float = 1e-10, lam_stop_value: float = 1e-15) \
        -> np.ndarray:
    """
    Creates test functions (functions that approximately satisfy A*x=0) using single level relaxation.

    Args:
        operator: an object that can calculate residuals (action A*x).
        rq: Rayleigh-quotient functor.
        method: iterative method functor (an iteration is a call to this method).
        x: test matrix initial approximation to the test functions.
        num_sweeps: number of sweeps to execute.
        print_frequency: print debugging convergence statements per this number of sweeps. None means no printouts.

    Returns:
        e: relaxed test matrix.
        conv_factor: asymptotic convergence factor (of the last iteration).
    """
    # Print the error and residual norm of the first test function.
    x0 = x[:, 0]
    x_norm = scaled_norm(x0)
    r_norm = scaled_norm(operator(x0))
    lam = rq(x0)
    lam_error = 1
    _LOGGER.debug("{:5d} |e| {:.8e} |r| {:.8e} lam {:.5f}".format(0, x_norm, r_norm, lam))
    # Run 'num_sweeps' relaxation sweeps.
    if print_frequency is None:
        print_frequency = num_sweeps // 10
    r_norm_history = [None] * (num_sweeps + 1)
    r_norm_history[0] = r_norm
    for i in range(1, num_sweeps + 1):
        r_norm_old = r_norm
        lam_old = lam
        lam_error_old = lam_error
        x = method(x)
        x0 = x[:, 0]
        x_norm = scaled_norm(x0)
        r_norm = scaled_norm(operator(x0))
        lam = rq(x0)
        lam_error = np.abs(lam - lam_old)
        if i % print_frequency == 0:
            _LOGGER.debug("{:5d} |e| {:.8e} |r| {:.8e} ({:.5f}) lam {:.5f} ({:.5f})".format(
                i, scaled_norm(x0), r_norm, r_norm / r_norm_old, lam, lam_error / lam_error_old))
        r_norm_history[i] = r_norm
        if r_norm < residual_stop_value or lam_error < lam_stop_value:
            r_norm_history = r_norm_history[:i + 1]
            break
    #return x, r_norm / r_norm_old
    # Average convergence factor over the last 5 steps.
    last_steps = min(5, len(r_norm_history) - 1)
    return x, (r_norm_history[-1] / r_norm_history[-last_steps-1]) ** (1 / last_steps)


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
