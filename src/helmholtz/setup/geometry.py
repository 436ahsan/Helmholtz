"""Functions related to the geometric locations of variables."""
import logging
import numpy as np

import helmholtz as hm

_LOGGER = logging.getLogger(__name__)


def coarse_locations(fine_location: np.ndarray, aggregate_size: int, num_components: int):
    """
    Calculate coarse-level variable locations given fine-level locations coarsened with aggregate_size/nc coarsening.
    At each aggregate center we have 'num_components' coarse variables.

    Args:
        fine_location: fine-level variable location.
        aggregate_size: size of window (aggregate).
        num_components: number of coarse variables per aggregate.

    Returns: array of coarse variable location.
    """
    return np.tile(np.add.reduceat(fine_location, np.arange(0, len(fine_location), aggregate_size)) / aggregate_size,
                   (num_components, 1)).T.flatten()


def geometric_neighbors(aggregate_size: int, num_components: int):
    """
    Returns the relative indices of the interpolation set of each fine variable in a window. Center neighbors are
    listed first, then neighboring aggregates.

    Args:
        aggregate_size: size of window (aggregate).
        num_components: number of coarse variables per aggregate. In 1D at the finest level, we know there are two principal
            components, so num_components=2 makes sense there.

    Returns: array of size w x {num_neighbors} of coarse neighbor indices (relative to fine variable indices) of each
        fine variable.
    """
    # Here we assume w points per aggregate and the same number mc of coarse vars per aggregate, but in general
    # aggregate sizes may vary.
    fine_var = np.arange(aggregate_size, dtype=int)

    # Index of neighboring coarse variable. Left neighboring aggregate for points on the left half of the window;
    # right for right.
    coarse_nbhr = np.zeros_like(fine_var)
    left = fine_var < aggregate_size // 2
    right = fine_var >= aggregate_size // 2
    coarse_nbhr[left] = -1
    coarse_nbhr[right] = 1

    # All nbhrs = central aggregate coarse vars + neighboring aggregate coarse vars.
    coarse_vars_center = np.tile(np.arange(num_components, dtype=int), (aggregate_size, 1))
    return np.concatenate((coarse_vars_center, coarse_vars_center + num_components * coarse_nbhr[:, None]), axis=1)


def geometric_neighbors_from_locations(fine_location: np.ndarray,
                                       coarse_location: np.ndarray,
                                       domain_size: float,
                                       aggregate_size: int,
                                       max_caliber: int = 6):
    """
    Calculate coarse-level variable locations given fine-level locations coarsened with aggregate_size/nc coarsening.
    At each aggregate center we have 'num_components' coarse variables.

    Args:
        fine_location: fine-level variable location.
        coarse_location: coarse-level variable location.
        domain_size: size of domain.
        aggregate_size: size of window (aggregate).
        max_caliber: number of neighbors to return.

    Returns: array of coarse variable location.
    """
    # Find nearest neighbors of each fine P^T*A point (pta_vars).
    # These are INDICES into the rap_vars array.
    nbhr = np.argsort(
        np.abs(hm.linalg.wrap_index_to_low_value(fine_location[:, None] - coarse_location, domain_size)), axis=1)
    return nbhr[:aggregate_size, :max_caliber]
