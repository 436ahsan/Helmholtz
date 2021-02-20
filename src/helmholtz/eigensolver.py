"""Eigensolver cycle business logic."""
import helmholtz as hm
import logging
import numpy as np

_LOGGER = logging.getLogger(__name__)

# x: initial guess. May be a vector of size n or a matrix of size n x m, where A is n x n.
#         Executes a relaxation V(nu_pre, nu_post) -cycle on A*x = 0.


def eigen_cycle(multilevel: hm.multilevel.Multilevel,
                cycle_index: float, nu_pre: int, nu_post: int, nu_coarsest: int,
                debug: bool = False, update_lam: str = "coarsest",
                relax_coarsest: int = 5,
                num_levels: int = None):
    if num_levels is None:
        num_levels = len(multilevel)
    processor = EigenProcessor(multilevel, nu_pre, nu_post, nu_coarsest, debug=debug, update_lam=update_lam,
                               relax_coarsest=relax_coarsest)
    return hm.cycle.Cycle(processor, cycle_index, num_levels)


class EigenProcessor(hm.processor.Processor):
    """
    Eigensolver cycle processor. Executes am eigensolver Cycle(nu_pre, nu_post, nu_coarsest) on A*x = lam*x.
    """
    def __init__(self, multilevel: hm.multilevel.Multilevel,
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
        self._sigma = np.ones(x.shape[1], )
        self._x_initial = [None] * len(self._multilevel)
        # Initialize finest-level quantities.
        self._x[l] = x
        self._b[l] = np.zeros_like(x)

    def process_coarsest(self, l):
        self._print_state(l, "initial")
        level = self._multilevel.level[l]
        for _ in range(self._nu_coarsest):
            for _ in range(self._relax_coarsest):
                self._x[l] = level.relax(self._x[l], self._b[l])
            if self._update_lam == "coarsest":
                # Update lambda + normalize only once per several relaxations if multilevel and updating lambda
                # at the coarsest level.
                self._x[l] = self._update_global_constraints(l, self._x[l])
        self._print_state(l, "coarsest ({})".format(self._nu_coarsest))

    def pre_process(self, l):
        # Execute at level L right before switching to the next-coarser level L+1.
        level = self._multilevel.level[l]
        self._print_state(l, "initial")
        self._relax(l, self._nu_pre)

        # Full Approximation Scheme (FAS).
        coarse_level = self._multilevel.level[l + 1]
        xc_initial = coarse_level.restrict(x)
        bc = coarse_level.restrict(b - level.operator(x)) + coarse_level.operator(xc_initial)
        sigma_c = sigma - level.normalization(x) + coarse_level.normalization(xc_initial)

    def post_process(self, l):
        coarse_level = self._multilevel.level[l + 1]
        x += coarse_level.interpolate(xc - xc_initial)
        self._print_state("correction {}".format(nu_pre))

        # Executes at level L right before switching to the next-finer level L-1.
        self._relax(l, self._nu_post)

    def _relax(self, l, num_sweeps):
        level = self._multilevel.level[l]
        if num_sweeps > 0:
            for _ in range(num_sweeps):
                self._x[l] = level.relax(self._x[l], self._b[l])
            self._print_state(l, "relax {}".format(num_sweeps))

    def post_cycle(self, l):
        # Executes at the finest level L at the end of the cycle. A hook.
        # TODO(orenlivne): add Ritz projection.
        pass

    def result(self, l):
        # Returns the cycle result X at level l. Normally called by Cycle with l=finest level.
        return self._x[l]

    def _print_state(self, level_ind, title):
        level = self._multilevel.level[level_ind]
        if self._debug:
            _LOGGER.debug("{:<5d}    {:<15}    {:.4e}    {:.4e}    {:.8f}".format(
                level_ind, title, scaled_norm(b[:, 0] - level.operator(x[:, 0])),
                np.abs(sigma - level.normalization(x))[0], level.global_params.lam))

    def _update_global_constraints(self, l, x):
        """
        Updates lambda + normalize at level 'level'.
        Args:
            x:

        Returns:
            Updated x. Global lambda also updated to the mean of RQ of all test functions.
        """
        level = self._multilevel.level[l]
        b = self._b[l]
        eta = level.normalization(x)
        # TODO(orenlivne): vectorize the following expressions.
        for i in range(x.shape[1]):
            x[:, i] *= (self._sigma[i] / eta[i]) ** 0.5
        # TODO(orenlivne): replace by multiple eigenvalue Gram Schmidt.
        level.global_params.lam = np.mean([level.rq(x[:, i], b[:, i]) for i in range(x.shape[1])])
        return x
