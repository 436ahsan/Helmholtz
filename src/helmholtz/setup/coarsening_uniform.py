"""coarsening (R) construction routines. Based on SVD on an aggregate."""
import logging
import numpy as np
import scipy.sparse
from numpy.linalg import svd
from typing import Generator, List, Tuple

import helmholtz as hm

_LOGGER = logging.getLogger(__name__)


def create_coarsening_domain_uniform(x, aggregate_size,
                                     cycle_index: float = 1, cycle_coarse_level_work_bound: float = 0.7,) -> \
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
    starts = get_aggregate_starts(domain_size, aggregate_size)
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
                                             shape=(r_last.shape[0], r_last.shape[1] + offset)).todense()
            # Merge the two.
            r = scipy.sparse.vstack((r, r_last))
        yield r, mean_energy_error[nc - 1]


def get_optimal_coarsening(level, x, aggregate_size_values, nu_values, max_conv_factor):
    """
    Returns a coarsening matrix (R) on the non-repetitive domain, which maximizes mock cycle efficiency over aggregate
    size and # principal components (coarse vars per aggregate).

    Args:
        level:
        x:
        aggregate_size_values:
        nu_values:
        max_conv_factor:

    Returns:

    """
    # Generates coarse variables (R) on the non-repetitive domain.
    result = [(aggregate_size, nc, r, mean_energy_error)
              for aggregate_size in aggregate_size_values
              for nc, (r, mean_energy_error) in enumerate(
            hm.setup.coarsening_uniform.create_coarsening_domain_uniform(x, aggregate_size), 1)]
    aggregate_size = np.array([item[0] for item in result])
    nc = np.array([item[1] for item in result])
    r_values = np.array([item[2] for item in result])
    mean_energy_error = np.array([item[3] for item in result])
    # Note: can be derived from cycle index.
    mock_conv_factor = np.array([[hm.setup.auto_setup.mock_cycle_conv_factor(level, r, nu) for nu in nu_values]
                                 for r in r_values])
    coarsening_ratio = np.array([r.shape[0] / r.shape[1] for r in r_values])
    work = nu_values[None, :] / (1 - coarsening_ratio[:, None])
    efficiency = mock_conv_factor ** (1 / work)
    i, j = np.where(mock_conv_factor <= max_conv_factor)
    candidate = np.vstack((
        i,
        aggregate_size[i],
        nc[i],
        coarsening_ratio[i],
        mean_energy_error[i],
        nu_values[j],
        mock_conv_factor[mock_conv_factor <= max_conv_factor],
        work[mock_conv_factor <= max_conv_factor],
        efficiency[mock_conv_factor <= max_conv_factor]
    )).transpose()
    print(candidate)
    best_index = np.argmin(candidate[:, -1])
    i, aggregate_size, nc, cr, mean_energy_error, nu, mock_conv, mock_work, mock_efficiency = candidate[best_index]
    return r_values[int(i)], int(aggregate_size), int(nc), cr, mean_energy_error, nu, mock_conv, mock_work, \
        mock_efficiency


def get_aggregate_starts(domain_size, aggregate_size):
    """
    Returns the list of aggregate starts, for a domain size and fixed aggregate size over the entire domain. The last
    two aggregates overlap if the domain size is not divisible by the aggregate size.
    Args:
        domain_size: domain size.
        aggregate_size: aggregate sixe.

    Returns: list of aggregate start indices.
    """
    return list(range(0, domain_size, aggregate_size)) if domain_size % aggregate_size == 0 else \
        list(range(0, domain_size - aggregate_size, aggregate_size)) + [domain_size - aggregate_size]
