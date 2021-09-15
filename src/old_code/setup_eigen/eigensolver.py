"""Eigensolver cycle business logic."""
import logging
import numpy as np

import helmholtz as hm
from helmholtz.linalg import scaled_norm

_LOGGER = logging.getLogger(__name__)

# x: initial guess. May be a vector of size n or a matrix of size n x m, where A is n x n.
#         Executes a relaxation V(nu_pre, nu_post) -cycle on A*x = 0.


def eigen_cycle(multilevel: hm.hierarchy.multilevel.Multilevel,
                cycle_index: float, nu_pre: int, nu_post: int, nu_coarsest: int,
                debug: bool = False, update_lam: str = "coarsest",
                relax_coarsest: int = 5,
                num_levels: int = None, finest: int = 0):
    if num_levels is None:
        num_levels = len(multilevel)
    processor = EigenProcessor(multilevel, nu_pre, nu_post, nu_coarsest, debug=debug, update_lam=update_lam,
                               relax_coarsest=relax_coarsest)
    return hm.hierarchy.cycle.Cycle(processor, cycle_index, num_levels, finest=finest)


class EigenProcessor(hm.hierarchy.processor.Processor):
    """
    Eigensolver cycle processor. Executes am eigensolver Cycle(nu_pre, nu_post, nu_coarsest) on A*x = lam*x.
    """
    def __init__(self, multilevel: hm.hierarchy.multilevel.Multilevel,
                 nu_pre: int, nu_post: int, nu_coarsest: int,
                 debug: bool = False, update_lam: str = "coarsest",
                 relax_coarsest: int = 5) -> np.array:
        """
        Args:
            multilevel: multilevel hierarchy to use in the cycle.
            nu_pre: number of relaxation sweeps at a level before visiting coarser levels.
            nu_post: number of relaxation sweeps at a level after visiting coarser levels.
            nu_coarsest: number of relaxation sweeps to run at the coarsest level.
            debug: print logging debugging printouts or not.
        """
        self._multilevel = multilevel
        self._nu_pre = nu_pre
        self._nu_post = nu_post
        self._nu_coarsest = nu_coarsest
        self._relax_coarsest = relax_coarsest
        self._debug = debug
        # TODO(orenlivne): remove once we have Ritz in place.
        self._update_lam = update_lam

        self._x = None
        self._lam = None
        self._x_initial = None
        self._b = None
        self._sigma = None

    def initialize(self, l, num_levels, initial_guess):
        if self._debug:
            _LOGGER.debug("-" * 80)
            _LOGGER.debug("{:<5}    {:<15}    {:<10}    {:<10}    {:<10}".format(
                "Level", "Operation", "|R|", "|R_norm|", "lambda"))
        x, lam = initial_guess
        # Allocate quantities at all levels.
        self._x = [None] * len(self._multilevel)
        self._b = [None] * len(self._multilevel)
        self._sigma = [None] * len(self._multilevel)
        self._x_initial = [None] * len(self._multilevel)
        # Initialize finest-level quantities.
        self._x[l] = x
        self._lam = lam
        self._b[l] = np.zeros_like(x)
        self._sigma[l] = np.ones(x.shape[1], )

    def process_coarsest(self, l):
        self._print_state(l, "initial")
        level = self._multilevel[l]
        for _ in range(self._nu_coarsest):
            for _ in range(self._relax_coarsest):
                self._x[l] = level.relax(self._x[l], self._b[l], self._lam)
            if self._update_lam == "coarsest":
                # Update lambda + normalize only once per several relaxations if multilevel and updating lambda
                # at the coarsest level.
                self._x[l], self._lam = self._update_global_constraints(l, self._x[l])
        self._print_state(l, "coarsest ({})".format(self._nu_coarsest))

    def pre_process(self, l):
        # Execute at level L right before switching to the next-coarser level L+1.
        level = self._multilevel[l]
        self._print_state(l, "initial")
        self._relax(l, self._nu_pre)

        # Full Approximation Scheme (FAS).
        lc = l + 1
        coarse_level = self._multilevel[lc]
        x, lam = self._x[l], self._lam
        xc_initial = coarse_level.coarsen(x)
        self._x_initial[lc] = xc_initial
        self._x[lc] = xc_initial
        self._b[lc] = coarse_level.restrict(self._b[l] - level.operator(x, lam)) + \
                      coarse_level.operator(xc_initial, lam)
        self._sigma[lc] = self._sigma[l] - level.normalization(x) + coarse_level.normalization(xc_initial)

    def post_process(self, l):
        lc = l + 1
        coarse_level = self._multilevel[lc]
        self._x[l] += coarse_level.interpolate(self._x[lc] - self._x_initial[lc])
        self._print_state(l, "correction")

        # Executes at level L right before switching to the next-finer level L-1.
        self._relax(l, self._nu_post)

    def _relax(self, l, num_sweeps):
        level = self._multilevel[l]
        if num_sweeps > 0:
            for _ in range(num_sweeps):
                self._x[l] = level.relax(self._x[l], self._b[l], self._lam)
            self._print_state(l, "relax {}".format(num_sweeps))

    def post_cycle(self, l):
        # Executes at the finest level L at the end of the cycle. A hook.
        level = self._multilevel[l]
        # # Exclude lam term from action here, so it's just A*x.
        # action = lambda x: level.stiffness_operator(x)
        # self._x[l], lam_ritz = hm.linalg.ritz(self._x[l], action)
        # # TODO(orenlivne): remove this once we start calculating multiple lambda.
        # self._lam = lam_ritz[0]

    def result(self, l):
        # Returns the cycle result X at level l. Normally called by Cycle with l=finest level.
        return self._x[l], self._lam

    def _print_state(self, l, title):
        level = self._multilevel[l]
        if self._debug:
            x = self._x[l]
            b = self._b[l]
            _LOGGER.debug("{:<5d}    {:<15}    {:.4e}    {:.4e}    {:.8f}".format(
                l, title, scaled_norm(b[:, 0] - level.operator(x[:, 0], self._lam)),
                np.abs(self._sigma[l] - level.normalization(x))[0], self._lam))

    def _update_global_constraints(self, l, x):
        """
        Updates lambda + normalize at level 'level'.
        Args:
            x:

        Returns:
            Updated x. Global lambda also updated to the mean of RQ of all test functions.
        """
        level = self._multilevel[l]
        b = self._b[l]
        sigma = self._sigma[l]
        eta = level.normalization(x)
        # TODO(orenlivne): vectorize the following expressions.
        for i in range(x.shape[1]):
            x[:, i] *= (sigma[i] / eta[i]) ** 0.5
        # TODO(orenlivne): replace by multiple eigenvalue Gram Schmidt.
        lam = np.mean([level.rq(x[:, i], b[:, i]) for i in range(x.shape[1])])
        return x, lam
