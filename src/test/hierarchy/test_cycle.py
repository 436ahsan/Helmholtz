import numpy as np
import pytest

import helmholtz as hm


_EPS = 1e-15


class TestCycle:

    @pytest.mark.parametrize("cycle_index", [1, 2, 3, 4, 1.2, 2.1])
    def test_num_level_visits_follows_cycle_index(self, cycle_index):
        """Tests that cycle-index-g-cycle counters behave like g^(l-1) to tolerenace tol.
         Here l=level index (0=finest).
        """
        num_levels = 7
        tol = 1e-14 if isinstance(cycle_index, int) else 0.2
        processor = _LevelVisitCounter()
        cycle = hm.hierarchy.cycle.Cycle(processor, cycle_index, num_levels)

        cycle.run(0)

        ratio = 1. / _factors(processor.num_visits[:-1])
        weighted_error = hm.linalg.scaled_norm((ratio - cycle_index) * np.arange(num_levels - 2))
        assert weighted_error < tol


def _factors(a):
    a[a < _EPS] = _EPS
    return np.exp(np.diff(-np.log(a)))


class _LevelVisitCounter(hm.hierarchy.processor.Processor):
    def __init__(self):
        self.num_visits = None

    def initialize(self, l, num_levels, initial_guess):
        self.num_visits = np.array([0] * num_levels)

    def pre_process(self, l):
        self.num_visits[l] += 1

    def post_process(self, l):
        pass

    def process_coarsest(self, l):
        # Run at the coarsest level L.
        self.num_visits[l] += 1
