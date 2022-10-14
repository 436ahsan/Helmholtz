import logging
import sys
import numpy as np
import pytest

import helmholtz as hm
import helmholtz.analysis

logger = logging.getLogger("nb")


class TestIdealTvInterpolation:
    def setup_method(self):
        """Fixed random seed for deterministic results."""
        np.set_printoptions(precision=6, linewidth=1000)
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)-8s %(message)s",
                            datefmt="%a, %d %b %Y %H:%M:%S")
        np.random.seed(1)

    def test_laplace_ideal_vectors_weighted_ls_yields_good_cycle(self):
        # Domain size.
        n = 96
        discretization = "3-point"
        kh = 0
        repetitive = True
        # Test vectors.
        num_examples = 2

        aggregate_size = 2  # 4
        num_components = 1  # 2

        # Interpolation parameters.
        interpolation_method = "ls"
        fit_scheme = "plain"
        weighted = True
        neighborhood = "extended"  # "aggregate" # "extended"
        num_test_examples = 5
        caliber = 2

        # Generate test vectors.
        a = hm.linalg.helmholtz_1d_operator(kh, n)
        level = hm.setup.hierarchy.create_finest_level(a)
        level.location = np.arange(level.size)
#        x = hm.setup.auto_setup.get_test_matrix(a, nu, num_examples=num_examples)
        x, _ = hm.analysis.ideal.ideal_tv(level.a, num_examples)

        r, s = hm.repetitive.locality.create_coarsening(x, aggregate_size, num_components, normalize=True)
        R = r.tile(level.size // aggregate_size)

        p = hm.setup.auto_setup.create_interpolation(
            x, a, R, level.location, n, interpolation_method, aggregate_size=aggregate_size, num_components=num_components,
            neighborhood=neighborhood, repetitive=repetitive, target_error=0.1,
            caliber=caliber, fit_scheme=fit_scheme, weighted=weighted)
        multilevel = hm.repetitive.locality.create_two_level_hierarchy(
            kh, discretization, n, R, p, p.T, aggregate_size, num_components)

        ac = multilevel[1].a
        fill_in_factor = (ac.nnz / multilevel[0].a.nnz) * (multilevel[0].a.shape[0] / ac.shape[0])
        assert fill_in_factor == pytest.approx(1.67, 1e-2)

        nu = 4
        y, conv_factor = hm.repetitive.locality.two_level_conv_factor(multilevel, nu, print_frequency=1,
                                                                      residual_stop_value=1e-9)
        assert conv_factor == pytest.approx(0.141, 1e-2)
