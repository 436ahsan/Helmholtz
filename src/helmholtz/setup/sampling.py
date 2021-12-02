"""Test function sampling, for repetitive problems."""
import helmholtz as hm
import numpy as np


def get_disjoint_windows(x, xc, r, aggregate_size: int, num_components: int, num_windows: int):
    """
    Samples windows of test functions.

    No need to use all windows of the domain in the least-squares fit. It's sufficient to use
    than the interpolation caliber. Using 12 because we need (train, val, test) for LS fit, each of which is
    say 4 times the caliber to be safe.

    :param x: fine-level test matrix.
    :param xc: coarse-level test matrix.
    :param r: fine-level residual matrix A*x.
    :param aggregate_size: aggregate size.
    :param num_components: number of coarse variables per aggregate.
    :param num_windows: #windows to use.
    :return:
    - Windows of fine-level test functions 'x' on disjoint aggregates (e.g., points 0..3, 4..7, etc. for
    aggregate_size = 4).
    - Windows of coarse-level test functions on the same aggregates.
    - Windows of local residual norms corresponding to the 'x' windows.
    """
    # Create windows of 'x'.
    x_disjoint_aggregate_t = hm.linalg.get_windows_by_index(x, np.arange(aggregate_size), aggregate_size, num_windows)

    # Create corresponding windows of 'xc'. Note: we are currently concatenating the entire coarse domain 'num_windows'
    # times. This is not necessary if neighbor computation is done here and not inside create_interpolation(). For
    # now, we keep create_interpolation() general and do the nbhr computation there.
    # TODO(orenlivne): reduce storage here using smart periodic indexing or calculate the nbhr set here first
    # and only pass windows of nbhr values to create_interpolation.
    xc_disjoint_aggregate_t = hm.linalg.get_windows_by_index(xc, np.arange(xc.shape[0]), num_components, num_windows)

    # Create local residual norms corresponding to the 'x'-windows. 3 * aggregate_size domain for norm is good for 1D.
    # TODO(orenlivne): in graph problems, replace residual_window_size by aggregate_size + sum of its
    #  residual_window_size aggregate sizes.
    r_norm_disjoint_aggregate_t = residual_norm_windows(r, 3 * aggregate_size, aggregate_size, num_windows)

    return x_disjoint_aggregate_t, xc_disjoint_aggregate_t, r_norm_disjoint_aggregate_t


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
