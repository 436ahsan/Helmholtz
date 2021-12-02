"""Test function sampling, for repetitive problems."""
import helmholtz as hm
import numpy as np


def residual_norm_windows(r, residual_window_size, aggregate_size, num_windows):
    # Each residual window is centered at the center of the aggregate = offset + aggregate_size // 2, and
    # extends ~ 0.5 * residual_window_size in each direction. Then the scaled L2 norm of window is calculated.
    window_start = aggregate_size // 2 - (residual_window_size // 2)

    r_windows = hm.linalg.get_windows_by_index(
        r, np.arange(window_start, window_start + residual_window_size), aggregate_size, num_windows)
    r_norm_disjoint_aggregate_t = np.linalg.norm(r_windows, axis=1) / residual_window_size ** 0.5

    # In principle, each point in the aggregate should have a slightly shifted residual window, but we just take the
    # same residual norm for all points for simplicity. Should not matter much.
    return np.tile(r_norm_disjoint_aggregate_t[:, None], (aggregate_size, ))
