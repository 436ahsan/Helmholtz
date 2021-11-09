"""Test function sampling, for repetitive problems."""
import helmholtz as hm
import numpy as np


def get_disjoint_windows(x, xc, r, aggregate_size, nc, max_caliber):
    """
    Samples windows of test functions.

    No need to use all windows of the domain in the least-squares fit. It's sufficient to use
    than the interpolation caliber. Using 12 because we need (train, val, test) for LS fit, each of which is
    say 4 times the caliber to be safe.

    :param x: fine-level test matrix.
    :param xc: coarse-level test matrix.
    :param r: fine-level residual matrix A*x.
    :param aggregate_size: aggregate size.
    :param nc: number of coarse variables per aggregate.
    :param max_caliber: maximum caliber to consider. #windows is proportional to it.
    :return:
    - Windows of fine-level test functions 'x' on disjoint aggregates (e.g., points 0..3, 4..7, etc. for
    aggregate_size = 4).
    - Windows of coarse-level test functions on the same aggregates.
    - Windows of local residual norms corresponding to the 'x' windows.
    """
    domain_size, num_test_functions = x.shape
    num_aggregates = int(np.ceil(domain_size / aggregate_size))
    num_windows = max(np.minimum(num_aggregates, (12 * max_caliber) // num_test_functions), 1)
    #    print("max_caliber", max_caliber, "num_test_functions", num_test_functions, "num_windows",
    #    num_windows, "num_aggregates", num_aggregates)

    # Create windows of 'x'.
    x_disjoint_aggregate_t = _get_disjoint_windows(x, aggregate_size, num_windows)

    # Create corresponding windows of 'xc'. Note: we are currently concatenating the entire coarse domain 'num_windows'
    # times. This is not necessary if neighbor computation is done here and not inside create_interpolation(). For
    # now, we keep create_interpolation() general and do the nbhr computation there.
    # TODO(orenlivne): reduce storage here using smart periodic indexing or calculate the nbhr set here first
    # and only pass windows of nbhr values to create_interpolation.
    xc_disjoint_aggregate_t = np.concatenate(
        tuple(hm.linalg.get_window(xc, nc * offset, nc * num_aggregates)
              for offset in range(num_windows)), axis=1).transpose()

    # Create local residual norms corresponding to the 'x'-windows.
    # TODO(orenlivne): in graph problems, replace residual_window_size by aggregate_size + sum of its residual_window_size
    # aggregate sizes.
    r_norm_disjoint_aggregate_t = _residual_norm_windows(r, aggregate_size, num_windows)

    return x_disjoint_aggregate_t, xc_disjoint_aggregate_t, r_norm_disjoint_aggregate_t


def _get_disjoint_windows(x, aggregate_size, num_windows):
    return np.concatenate(
        tuple(hm.linalg.get_window(x, aggregate_size * offset, aggregate_size)
              for offset in range(num_windows)),
        axis=1).transpose()


def _residual_norm_windows(r, aggregate_size, num_windows):
    residual_window_size = 3 * aggregate_size  # Good for 1D.
    residual_window_offset = -(residual_window_size // 2)
    # Each residual window is centered at the center of the aggregate = offset + aggregate_size // 2, and
    # extends ~0.5 * residual_window_size in each direction. Then the scaled L2 norm of window is calculated.
    r_norm_disjoint_aggregate_t = np.concatenate(tuple(np.linalg.norm(
        hm.linalg.get_window(r,
                             (offset + aggregate_size // 2 + residual_window_offset) % r.shape[0],
                             residual_window_size),
        axis=0) for offset in range(num_windows))) / residual_window_size ** 0.5
    # In principle, each point in the aggregate should have a slightly shifted residual window, but we just take the
    # same residual norm for all points for simplicity. Should not matter much.
    return np.tile(r_norm_disjoint_aggregate_t[:, None], (aggregate_size, ))
