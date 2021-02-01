"""Bootstrap AMG processes that generate test functions with low Helmholtz residuals on a periodic domain."""
import logging
from typing import Tuple

import numpy as np
import scipy.sparse

import helmholtz as hm

_LOGGER = logging.getLogger(__name__)


def generate_test_functions(a: scipy.sparse.spmatrix, num_growth_steps: int,
                            growth_factor: int = 2,
                            num_bootstrap_steps: int = 1,
                            aggregate_size: int = 4) -> Tuple[np.ndarray, hm.multilevel.Multilevel]:
    """
    Creates low-residual test functions and multilevel hierarchy on a large domain from the operator on a small window
    (the coarsest domain). This is similar to a full multigrid algorithm.

    Args:
        a: the operator on the coarsest domain.
        num_growth_steps: number of steps to increase the domain.
        growth_factor: by how much to increase the domain size at each growth step.
        num_bootstrap_steps: number of bootstrap steps to perform on each domain.
        aggregate_size: aggregate size = #fine vars per aggregate

    Returns:
        x: test matrix on the largest (final) domain.
        multilevel: multilevel hierarchy on the largest (final) domain.
    """
    # Initialize test functions (to random) and hierarchy at coarsest level.
    multilevel = hm.multilevel.Multilevel()
    level = hm.multilevel.Level.create_finest_level(a)
    multilevel.level.append(level)
    x = hm.multilevel.random_test_matrix(domain_shape, num_examples=num_examples)
    # Bootstrap at the current level.
    for i in range(num_bootstrap_steps):
        x, multilevel = bootstap(x, multilevel)

    for l in range(num_growth_steps):
        _LOGGER.info("Growing domain to level {}, size {}".format(l, growth_factor * x.shape[0]))
        # Tile solution and hierarchy on the twice larger domain.
        x = hm.linalg.tile_csr_matrix(x, growth_factor)
        multilevel = multilevel.tile_csr_matrix(growth_factor)
        # Bootstrap at the current level.
        for i in range(num_bootstrap_steps):
            x, multilevel = bootstap(x, multilevel, aggregate_size=aggregate_size)

    return x, multilevel


def bootstap(x, multilevel, aggregate_size: int = 4, nc: int = 2):
    """
    Improves test functions and a multilevel hierarchy on a fixed-size domain by bootstrapping.
    Args:
        x: test matrix.
        multilevel: multilevel hierarchy.
        aggregate_size: aggregate size = #fine vars per aggregate.
        nc: number of coarse variables per aggregate. In 1D at the finest level, we know there are two principal
            components, so nc=2 makes sense there.

    Returns:
        improved x, multilevel hierarchy with the same number of levels.
    """
    # At the finest level, use a multilevel cycle to improve x.
    level = multilevel.level[0]
    num_levels = len(multilevel)
    b = np.zeros_like(x)
    # TODO(orenlivne): update parameters of relaxation cycle to reasonable values if needed.
    relax_cycle = lambda x: multilevel.relax(x, 2, 2, 4) if num_levels > 1 else level.relax
    x, _ = hm.multilevel.relax_test_matrix(level.operator, relax_cycle, x, 100)

    # Recreate all coarse levels. One down-pass, relaxing at each level, hopefully starting from improved x so the
    # process improves all levels.
    # TODO(orenlivne): add nested bootstrap cycles if needed.
    new_multilevel = hm.multilevel.Multilevel()
    new_multilevel.append(level)
    for l in range(num_levels):
        _LOGGER.info("Coarsening level {}->{}".format(l - 1, l))
        level = create_coarse_level(level, x, aggregate_size=aggregate_size, num_examples=num_examples, nc=nc)
        new_multilevel.append(level)
        x = level.r.dot(x)
        x = hm.multilevel.relax_test_matrix(
            level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps, print_frequency=print_frequency)
        multilevel.level.append(coarse_level)
        return coarse_level

    return x, new_multilevel


def create_coarse_level(x, domain_size: int, aggregate_size: int, threshold: float = 0.1, caliber: int = 2):
    """
    Coarsens a level.
    Args:
        x: fine-level test matrix.
        domain_size: #gridpoints in fine level.
        aggregate_size: aggregate size = #fine vars per aggregate
        threshold: relative reconstruction error threshold. Determines nc.
        caliber: interpolation caliber.

    Returns: coarse level obtained by fitting P, R to x.
    """
    # TODO(oren): generalize the domain to the d-dimensional case. For now assuming 1D only.
    assert domain_size % aggregate_size == 0, \
        "Aggregate shape must divide the domain shape in every dimension"
    # assert all(ni % ai == 0 for ni, ai in zip(domain_shape, aggregate_shape)), \
    #     "Aggregate shape must divide the domain shape in every dimension"
    num_aggregates = domain_size // aggregate_size

    x_aggregate_t = x[:aggregate_size].transpose()
    r, s = hm.restriction.create_restriction(x_aggregate_t, threshold)
    _LOGGER.debug("Singular values {}, nc {}".format(s, r.shape[0]))
    xc = r.dot(x)
    p = hm.interpolation.create_interpolation(
        r.asarray(threshold), x_aggregate_t, xc.transpose(), domain_size, nc, caliber)
    return hm.multilevel.Level.create_coarse_level(ac, r, p)
