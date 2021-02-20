"""Cycle processor API. Executes business logic at each cycle level."""


class Processor:
    """
    A cycle processor base class. Executes business logic at each cycle level.
    """
    def initialize(self, l, num_levels, initial_guess):
        # Runs at the beginning of an NUMLEVELS-level cycle at the finest
        # level L. INITIALGUESS is the initial value of the iterate passed
        # into the cycle. The RESULT field retrieves the iterate at the end
        # of the cycle.
        pass

    def process_coarsest(self, l):
        # Run at the coarsest level L.
        pass

    def pre_process(self, l):
        # Execute at level L right before switching to the next-coarser
        # level L+1.
        pass

    def post_process(self, l):
        # Executes at level L right before switching to the next-finer level
        # L-1.
        pass

    def post_cycle(self, l):
        # Executes at the finest level L at the end of the cycle. A hook.
        pass

    def result(self, l):
        # Returns the cycle result X at level l. Normally called by Cycle with l=finest level.
        return None
