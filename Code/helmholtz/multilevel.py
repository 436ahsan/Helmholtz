"""Multilevel solver (producer of low-residual test functions of the Helmholtz operator."""
import helmholtz as hm
import logging
import numpy as np
from typing import Tuple
from helmholtz.linalg import scaled_norm
from numpy.linalg import norm


_LOGGER = logging.getLogger(__name__)


class Multilevel:
    """The multilevel hierarchy. Contains a sequence of levels."""
    def __init__(self):
        self.level = []

    def __len__(self):
        return len(self.level)


class Level:
    """A single level in the multilevel hierarchy."""
    def __init__(self, a, r=None, p=None):
        self.a = a
        self.r = r
        self.p = p
        self._relaxer = hm.kaczmarz.KaczmarzRelaxer(a)

    def print(self):
        _LOGGER.info("a = \n" + str(self.a.toarray()))
        if self.r is not None:
            _LOGGER.info("r = \n" + str(self.r.toarray()))
        if self.p is not None:
            _LOGGER.info("p = \n" + str(self.p.toarray()))

    def operator(self, x: np.array) -> np.array:
        """
        Returns the operator action A*x..
        Args:
            x: vector of size n or a matrix of size n x m, where A is n x n.

        Returns:
            A*x.
        """
        return self.a.dot(x)

    def relax(self, x: np.array) -> np.array:
        """
        Executes a relaxation sweep on A*x = 0 at this level.
        Args:
            x: initial guess. May be a vector of size n or a matrix of size n x m, where A is n x n.

        Returns:
            x after relaxation.
        """
        return self._relaxer.step(x)

    def create_relaxed_test_matrix(self,
                                   window_shape: Tuple[int],
                                   num_examples: int,
                                   num_sweeps: int = 30) -> np.ndarray:
        """
        Creates test functions (functions that approximately satisfy A*x=0) using single level relaxation.

        Args:
            window_shape: domain size (#gridpoints in each dimension).
            num_examples: number of test functions to generate.
            num_sweeps: number of sweeps to execute.

        Returns:
            e: window_size x num_examples test matrix.
        """
        if num_examples is None:
            # By default, use more test functions than gridpoints so we have a sufficiently large test function sample.
            num_examples = 4 * np.prod(window_shape)

        # Start from random[-1,1] guess for each function.
        e = 2 * np.random.random(window_shape + (num_examples,)) - 1
        # Print the error and residual norm of the first test function.
        # A poor way of getting the last "column" of the tensor e.
        e0 = e.reshape(-1, e.shape[-1])[:, 0].reshape(e.shape[:-1])
        _LOGGER.debug("{:5d} |e| {:.8e} |r| {:.8e}".format(0, scaled_norm(e0), scaled_norm(self.operator(e0))))

        # Run 'num_sweeps' relaxation sweeps.
        for i in range(1, num_sweeps + 1):
            e = self.relax(e)
            if i % (num_sweeps // 10) == 0:
                # A poor way of getting the last "column" of the tensor e.
                e0 = e.reshape(-1, e.shape[-1])[:, 0].reshape(e.shape[:-1])
                _LOGGER.debug("{:5d} |e| {:.8e} |r| {:.8e}".format(i, scaled_norm(e0), scaled_norm(self.operator(e0))))
            # Scale e to unit norm to avoid underflow, as we are calculating eigenvectors.
            e /= norm(e)
        return e
