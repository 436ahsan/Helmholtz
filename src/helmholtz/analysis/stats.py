import helmholtz as hm
import logging
import numpy as np
import pandas as pd
from numpy.linalg import norm

_LOGGER = logging.getLogger(__name__)


def compare_coarsening(level, nu,
                       domain_size: float,
                       aggregate_size: int, num_components: int,
                       calibers=(2, 3, 4), ideal_tv: bool = False,
                       num_examples: int = 5,
                       nu_values: np.ndarray = np.arange(1, 12),
                       interpolation_method: str = "ls",
                       fit_scheme: str = "ridge",
                       weighted: bool = False,
                       neighborhood: str = "extended",
                       repetitive: bool = False,
                       nu_coarsest: int = -1):
    # Generate initial test vectors.
    if ideal_tv:
        _LOGGER.info("Generating {} ideal TVs".format(num_examples))
        x, lam = hm.analysis.ideal.ideal_tv(level.a, num_examples)
    else:
        _LOGGER.info("Generating {} TVs with {} sweeps".format(num_examples, nu))
        x = hm.setup.auto_setup.get_test_matrix(level.a, nu, num_examples=num_examples)
        _LOGGER.info("RER {:.3f}".format(norm(level.a.dot(x)) / norm(x)))

    # Create coarsening.
    r, s = hm.repetitive.locality.create_coarsening(x, aggregate_size, num_components, normalize=False)
    r = r.tile(level.size // aggregate_size)
    xc = r.dot(x)

    # Mock cycle rates.
    mock_conv = [hm.setup.auto_setup.mock_cycle_conv_factor(level, r, nu) for nu in nu_values]

    # Interpolation by LS fitting for different calibers.
    p = dict((caliber, hm.setup.auto_setup.create_interpolation(
        x, level.a, r, level.location, domain_size, interpolation_method, aggregate_size=aggregate_size,
        num_components=num_components,
        neighborhood=neighborhood, repetitive=repetitive, target_error=0.1,
        caliber=caliber, fit_scheme=fit_scheme, weighted=weighted)) for caliber in calibers)

    # Symmetrizing restriction for high-order P.
    q = hm.repetitive.symmetry.symmetrize(r, level.a.dot(p[4]), aggregate_size, num_components)

    # Measure 2-level rates.
    # Each entry is (title, r, p, q).
    coarsening_types = [
        ("P^T A P caliber 2", r, p[2], p[2].T),
        ("P^T A P caliber 3", r, p[3], p[3].T),
        ("P^T A P caliber 4", r, p[4], p[4].T),
        ("R A P caliber 4", r, p[4], r),
        ("Q A P caliber 4", r, p[4], q),
    ]

    l2c = []
    a = level.a
    for _, r, p, q in coarsening_types:
        ml = hm.repetitive.locality.create_two_level_hierarchy_from_matrix(
            a, level.location, r, p, q, aggregate_size, num_components)
        ac = ml[1].a
        fill_in_factor = (ac.nnz / a.nnz) * (a.shape[0] / ac.shape[0])
        symmetry_deviation = np.max(np.abs(ac - ac.transpose()))
        two_level_conv = [hm.repetitive.locality.two_level_conv_factor(
            ml, nu, nu_coarsest=nu_coarsest, print_frequency=None)[1] for nu in nu_values]
        l2c.append([fill_in_factor, symmetry_deviation] + two_level_conv)

    data = np.array([[np.nan] * 2 + mock_conv] + l2c)
    all_conv = pd.DataFrame(np.array(data),
                            columns=("Fill-in", "Symmetry") + tuple(nu_values),
                            index=("Mock",) + tuple(item[0] for item in coarsening_types))
    return all_conv, r, p, q
