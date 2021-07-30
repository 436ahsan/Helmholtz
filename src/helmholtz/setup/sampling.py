"""Test function sampling, for repetitive problems."""
import helmholtz as hm
import numpy as np


def get_disjoint_windows(x, xc, aggregate_size, nc, max_caliber):
    """
     Creates windows of fine-level test functions on disjoint aggregates (e.g., points 0..3, 4..7, etc. for
    aggregate_size = 4). No need to use all windows of the domain in the least-squares fit. It's sufficient to use
    than the interpolation caliber. Using 12 because we need (train, val, test) for LS fit, each of which is
    say 4 times the caliber to be safe.
    """
    domain_size, num_test_functions = x.shape
    num_aggregates = int(np.ceil(domain_size / aggregate_size))
    num_windows = max(np.minimum(num_aggregates, (12 * max_caliber) // num_test_functions), 1)
#    print("max_caliber", max_caliber, "num_test_functions", num_test_functions, "num_windows", num_windows, "num_aggregates", num_aggregates)
    x_disjoint_aggregate_t = np.concatenate(
        tuple(hm.linalg.get_window(x, aggregate_size * offset, aggregate_size)
              for offset in range(num_windows)),
        axis=1).transpose()
    # Create corresponding windows of xc. Note: we are currently concatenating the entire coarse domain 'num_windows'
    # times. This is not necessary if neighbor computation is done here and not inside create_interpolation(). For
    # now, we keep create_interpolation() general and do the nbhr computation there.
    # TODO(orenlivne): reduce storage here using smart periodic indexing or calculate the nbhr set here first
    # and only pass windows of nbhr values to create_interpolation.
    num_coarse_vars = nc * num_aggregates
    xc_disjoint_aggregate_t = np.concatenate(tuple(hm.linalg.get_window(xc, nc * offset, num_coarse_vars)
                                                   for offset in range(num_windows)), axis=1).transpose()
    return x_disjoint_aggregate_t, xc_disjoint_aggregate_t
