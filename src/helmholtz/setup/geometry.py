"""Functions related to the geometric locations of variables."""
import logging
import numpy as np

import helmholtz as hm

_LOGGER = logging.getLogger(__name__)


def coarse_locations(fine_location: np.ndarray, aggregate_size: int, nc: int):
    """
    Calculate coarse-level variable locations given fine-level locations coarsened with aggregate_size/nc coarsening.
    At each aggregate center we have 'num_components' coarse variables.

    Args:
        fine_location: fine-levl variable location.
        aggregate_size: size of window (aggregate).
        nc: number of coarse variables per aggregate.

    Returns: array of coarse variable location.
    """
    return np.tile(np.add.reduceat(fine_location, np.arange(0, len(fine_location), aggregate_size)) / aggregate_size,
                   (nc, 1)).transpose().flatten()


def geometric_neighbors(aggregate_size: int, nc: int):
    """
    Returns the relative indices of the interpolation set of each fine variable in a window. Center neighbors are
    listed first, then neighboring aggregates.

    Args:
        aggregate_size: size of window (aggregate).
        nc: number of coarse variables per aggregate. In 1D at the finest level, we know there are two principal
            components, so nc=2 makes sense there.

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
    coarse_vars_center = np.tile(np.arange(nc, dtype=int), (aggregate_size, 1))
    return np.concatenate((coarse_vars_center, coarse_vars_center + nc * coarse_nbhr[:, None]), axis=1)
