"""A generic multi-level cycle."""
import logging
import numpy as np
import helmholtz.hierarchy.processor as hsc

_LOGGER = logging.getLogger(__name__)


class Cycle:
    """
    Runs a multilevel cycle. This class is responsible for the cycle's general control (switching between
    levels), leaving the business logic to a delegate processor object.
    """

    def __init__(self, processor: hsc.Processor, cycle_index, num_levels: int, finest: int = 0):
        """
        Creates an NUMLEVEL-level cycle at level FINEST with cycle index cycle_index that executes the
        business logic of PROCESSOR.

        Args:
            processor: executes level business logic.
            cycle_index: cycle index (array type means per level; scalar means uniform value for all levels).
            num_levels: number of levels in cycle.
            finest: Index of finest level to run cycles on.
        """
        # Coarsest level.
        # cycle levels
        # Finest level to run cycles on.
        self._processor = processor

        self._processor = processor
        self._cycle_index = [cycle_index] * (num_levels - 1) \
            if np.isscalar(cycle_index) and (num_levels is not None) else cycle_index
        self._num_levels = num_levels
        self._finest = finest
        self._coarsest = processor.coarsest if hasattr(processor, "coarsest") and processor.coarsest is not None \
            else (finest + num_levels - 1)

    def run(self, x):
        """
        Executes a cycle at the finest level.

        Args:
            x: current finest-level approximate solution, or a bundle (a tuple of x, its corresponding
                residual, eigenvalues, etc.)

        Returns: updated x (solution or bundle).
        """
        # Initialize state and level visitation counters: l = current level; k = next level to visit.
        L = self._coarsest
        num_visits = [0] * L
        p = self._processor
        f = self._finest
        gamma = [self._cycle_index] * (L - 1) if np.isscalar(self._cycle_index) else self._cycle_index
        # Inject pre-cycle iterate
        l = f
        p.initialize(l, self._num_levels, x)

        # Execute until we return to the finest level
        while True:
            # Compute the next level to process given the current level L
            # and the cycle visitation state (default
            # fractional-cycle-index algorithm). This hook can be
            # overridden by sub-classes.
            if l == L:
                # Coarsest level, go to next-finer level.
                k = l - 1
            else:
                max_visits = 1 if l == f else gamma[l] * num_visits[l - 1]
                k = (l + 1) if num_visits[l] < max_visits else (l - 1)
            if l == L:
                # Process coarsest level.
                p.process_coarsest(l)

            if k < f:
                break
            elif k > l:
                # Since we've just started visiting level l, increment its counter.
                num_visits[l] += 1
                # Go from level l to next-coarser level k.
                p.pre_process(l)
            else:
                # Go from level l to next-finer level k.
                p.post_process(k)
            # Update state
            l = k

        p.post_cycle(self._finest)
        # Retrieve post-cycle iterate.
        return p.result(self._finest)

    def _get_coarsest_level_index(self, coarsest_default):
        """Returns the index of the coarsest level."""
        return self._coarsest_override if self._coarsest_override > 0 else coarsest_default
