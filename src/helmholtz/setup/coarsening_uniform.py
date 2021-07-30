"""coarsening (R) construction routines. Based on SVD on an aggregate."""
import logging
import numpy as np
import scipy.sparse
from numpy.linalg import svd
from typing import Generator, List, Tuple

import helmholtz as hm

_LOGGER = logging.getLogger(__name__)


def create_coarsening_domain_uniform(x, aggregate_size, cycle_index: float = 1,
                                     cycle_coarse_level_work_bound: float = 0.7,
                                     repetitive: bool = False) -> \
        Generator[Tuple[scipy.sparse.csr_matrix, List[np.ndarray]], None, None]:
    """
    Creates the next coarse level's SVD coarsening operator R on a full domain (non-repetitive).
    Uses a fixed-size aggregate.

    Args:
        x: fine-level test matrix.
        cycle_index: cycle index of the cycle we are designing.
        cycle_coarse_level_work_bound: cycle_index * max_coarsening_ratio. Bounds the proportion of coarse level work
            in the cycle.
        aggregate_size: uniform aggregate size throughout the domain.
        repetitive: whether to exploit problem repetitiveness by creating a constant R stencil on all aggregates
            using windows from a single (or few) test vectors.

    Returns: a generator of (coarsening operator R, mean energy error over all aggregates), for all
        nc = 1..max_components (that satisfy the cycle work bound).
    """
    assert (0 <= cycle_coarse_level_work_bound <= 1) and (cycle_index > 0)
    domain_size = x.shape[0]
    # Maximum number of components to keep in each aggregate SVD.
    num_components = int(aggregate_size * cycle_coarse_level_work_bound / cycle_index)

    # Sweep the domain left to right; add an aggregate and its coarsening until we get to the domain end. Use a uniform
    # aggregate size; the last two aggregates will overlap of the domain size is not divisible by the aggregate size.
    # Find PCs of each aggregate.
    starts = hm.linalg.get_uniform_aggregate_starts(domain_size, aggregate_size)
    if repetitive:
        # Keep enough windows so that we have enough samples (4 * aggregate_size) an over-determined LS problem for R.
        x_aggregate_t = np.concatenate(
            tuple(hm.linalg.get_window(x, offset, aggregate_size)
                  for offset in range(max((4 * aggregate_size) // x.shape[1], 1))), axis=1).transpose()
        # Tile the same coarsening over all aggregates.
        aggregate_coarsening = hm.setup.coarsening.create_coarsening(x_aggregate_t, None, nc=num_components)
        svd_results = [aggregate_coarsening for _ in starts]
    else:
        svd_results = [
            hm.setup.coarsening.create_coarsening(x[start:start + aggregate_size].transpose(), None, nc=num_components)
            for start in starts
        ]
    r_aggregate_candidates = tuple(aggregate_svd_result[0].asarray() for aggregate_svd_result in svd_results)

    # Singular values, used for checking energy error in addition to the mock cycle criterion.
    s_aggregate = np.concatenate(tuple(aggregate_svd_result[1][None, :] for aggregate_svd_result in svd_results))
    mean_energy_error = np.mean(
        (1 - np.cumsum(s_aggregate ** 2, axis=1) / np.sum(s_aggregate ** 2, axis=1)[:, None]) ** 0.5,
        axis=0)

    # Merge all aggregate coarsening operators into R.
    for nc in range(1, num_components + 1):
        r_aggregate = [candidate[:nc] for candidate in r_aggregate_candidates]
        if domain_size % aggregate_size == 0:
            # Non-overlapping aggregates => R is block diagonal.
            r = scipy.sparse.block_diag(r_aggregate).tocsr()
        else:
            # Overlapping aggregate. Form the block-diagonal matrix except the last aggregate, then add it in.
            r = scipy.sparse.block_diag(r_aggregate[:-1]).tocsr()
            # Add columns to the matrix of the "interior" aggregates.
            r = scipy.sparse.csr_matrix((r.data, r.indices, r.indptr), (r.shape[0], domain_size))
            # Create a matrix for the boundary aggregate.
            r_last = scipy.sparse.csr_matrix(r_aggregate[-1])
            offset = starts[-1]
            r_last = scipy.sparse.csr_matrix((r_last.data, r_last.indices + offset, r_last.indptr),
                                             shape=(r_last.shape[0], domain_size)).todense()
            # Merge the two.
            r = scipy.sparse.vstack((r, r_last)).tocsr()
        yield r, mean_energy_error[nc - 1]


class UniformCoarsener:
    """
    Creates the next coarse level's SVD coarsening operator R on a full domain (non-repetitive).
    Uses a fixed-size aggregate and #PCs throughout the domain.
    """
    def __init__(self, level, x, aggregate_size_values, nu_values, cycle_index: float = 1,
                 cycle_coarse_level_work_bound: float = 0.7, repetitive: bool = False):
        """

        Args:
            level: level object containing the relaxation scheme.
            x: fine-level test matrix.
            aggregate_size_values: aggregate sizes to optimize over.
            nu_values: #sweep per cycle values to optimize over.
            cycle_index: cycle index of the cycle we are designing.
            cycle_coarse_level_work_bound: cycle_index * max_coarsening_ratio. Bounds the proportion of coarse level
                work in the cycle.
            repetitive: whether to exploit problem repetitiveness by creating a constant R stencil on all aggregates
                using windows from a single (or few) test vectors.
        """
        # Generates coarse variables (R) on the non-repetitive domain.
        self._result = [(aggregate_size, nc, r, mean_energy_error)
                  for aggregate_size in aggregate_size_values
                  for nc, (r, mean_energy_error) in enumerate(
                hm.setup.coarsening_uniform.create_coarsening_domain_uniform(
                    x, aggregate_size, cycle_index=cycle_index,
                    cycle_coarse_level_work_bound=cycle_coarse_level_work_bound,
                    repetitive=repetitive),
                1)]
        self._nu_values = nu_values
        r_values = np.array([item[2] for item in self._result])
        self._mock_conv_factor = np.array([[hm.setup.auto_setup.mock_cycle_conv_factor(level, r, nu)
                                            for nu in nu_values] for r in r_values])

    # TODO(oren): max_conv_factor can be derived from cycle index instead of being passed in.
    def get_coarsening_info(self, max_conv_factor):
        """
        Returns a table of coarsening matrix performance statistics vs. aggregate size and # principal components
        (coarse vars per aggregate).

        Args:
            max_conv_factor: max convergence factor to allow. NOTE: in principle, should be derived from cycle index.

        Returns:
            table of index into the _result array, aggregate_size, nc, cr, mean_energy_error, nu, mock_conv, mock_work,
            mock_efficiency.
        """
        aggregate_size = np.array([item[0] for item in self._result])
        nc = np.array([item[1] for item in self._result])
        r_values = np.array([item[2] for item in self._result])
        mean_energy_error = np.array([item[3] for item in self._result])
        coarsening_ratio = np.array([r.shape[0] / r.shape[1] for r in r_values])
        work = self._nu_values[None, :] / (1 - coarsening_ratio[:, None])
        efficiency = self._mock_conv_factor ** (1 / work)
        candidate = self._mock_conv_factor <= max_conv_factor
        i, j = np.where(candidate)
        candidate = np.vstack((
            i,
            aggregate_size[i],
            nc[i],
            coarsening_ratio[i],
            mean_energy_error[i],
            self._nu_values[j],
            self._mock_conv_factor[candidate],
            work[candidate],
            efficiency[candidate]
        )).transpose()
        return candidate
#        return np.array(candidate, [("i", "i4"), ("a", "i4"), ("nc", "i4"), ("cr", "f8"), ("Energy Error", "f8"),
#                                    ("nu", "f8"), ("conv", "f8"), ("work", "f8"), ("eff", "f8")])

    # TODO(oren): max_conv_factor can be derived from cycle index instead of being passed in.
    def get_optimal_coarsening(self, max_conv_factor):
        """
        Returns a coarsening matrix (R) on the non-repetitive domain, which maximizes mock cycle efficiency over
        aggregate size and # principal components (coarse vars per aggregate).

        Args:
            max_conv_factor: max convergence factor to allow. NOTE: in principle, should be derived from cycle index.

        Returns:
            Optimal R, aggregate_size, nc, cr, mean_energy_error, nu, mock_conv, mock_work, mock_efficiency.
        """
        candidate = self.get_coarsening_info(max_conv_factor)
        if candidate.size == 0:
            _LOGGER.info("Candidates coarsening")
            _LOGGER.info(self.get_coarsening_info(1.0))
            raise Exception("Could not find a coarsening whose mock cycle is below {:.2f}".format(max_conv_factor))
        best_index = np.argmin(candidate[:, -1])
        i, aggregate_size, nc, cr, mean_energy_error, nu, mock_conv, mock_work, mock_efficiency = candidate[best_index]
        return self._result[int(i)][2], int(aggregate_size), int(nc), cr, mean_energy_error, int(nu), mock_conv, \
            mock_work, mock_efficiency
