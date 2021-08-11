"""Multilevel relaxation cycle."""
import logging

import numpy as np

import helmholtz as hm
import helmholtz.hierarchy.multilevel as multilevel

_LOGGER = logging.getLogger(__name__)


def relax_cycle(multilevel: multilevel.Multilevel,
                cycle_index: float, nu_pre: int, nu_post: int, nu_coarsest: int,
                debug: bool = False, num_levels: int = None, finest: int = 0):
    if num_levels is None:
        num_levels = len(multilevel)
    processor = RelaxCycleProcessor(multilevel, nu_pre, nu_post, nu_coarsest, debug=debug)
    return hm.hierarchy.cycle.Cycle(processor, cycle_index, num_levels, finest=finest)


class RelaxCycleProcessor(hm.hierarchy.processor.Processor):
    """
    Relaxation cycle processor. Executes a Cycle(nu_pre, nu_post, nu_coarsest) on A*x = 0.
    """
    def __init__(self, multilevel: multilevel.Multilevel,
                 nu_pre: int, nu_post: int, nu_coarsest: int, debug: bool = False) -> np.array:
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
        self._debug = debug
        self._x = None
        self._x_initial = None
        self._b = None

    def initialize(self, l, num_levels, x):
        if self._debug:
            _LOGGER.debug("-" * 80)
            _LOGGER.debug("{:<5}    {:<15}    {:<10}    {:<10}".format("Level", "Operation", "|R|", "RER"))
        # Allocate quantities at all levels.
        self._x = [None] * len(self._multilevel)
        self._b = [None] * len(self._multilevel)
        self._x_initial = [None] * len(self._multilevel)
        # Initialize finest-level quantities.
        self._x[l] = x
        self._b[l] = np.zeros_like(x)

    def process_coarsest(self, l):
        self._print_state(l, "initial")
        level = self._multilevel[l]
        for _ in range(self._nu_coarsest):
            self._x[l] = level.relax(self._x[l], self._b[l])
        self._print_state(l, "coarsest ({})".format(self._nu_coarsest))

    def pre_process(self, l):
        # Execute at level L right before switching to the next-coarser level L+1.
        level = self._multilevel[l]
        self._print_state(l, "initial")
        self._relax(l, self._nu_pre)

        # Full Approximation Scheme (FAS).
        lc = l + 1
        coarse_level = self._multilevel[lc]
        x = self._x[l]
        xc_initial = coarse_level.coarsen(x)
        self._x_initial[lc] = xc_initial
        self._x[lc] = xc_initial
        self._b[lc] = coarse_level.restrict(self._b[l] - level.operator(x)) + coarse_level.operator(xc_initial)

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
                self._x[l] = level.relax(self._x[l], self._b[l])
            self._print_state(l, "relax {}".format(num_sweeps))

    def result(self, l):
        # Returns the cycle result X at level l. Normally called by Cycle with l=finest level.
        return self._x[l]

    def _print_state(self, level_ind, title):
        level = self._multilevel[level_ind]
        if self._debug:
            x = self._x[level_ind]
            r_norm = scaled_norm(b[:, 0] - level.operator(x[:, 0]))
            x_norm = scaled_norm(x[:, 0])
            _LOGGER.debug("{:<5d}    {:<15}    {:.4e}    {:.4e}".format(
                level_ind, title, r_norm, r_norm / x_norm))
