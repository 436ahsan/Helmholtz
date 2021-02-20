"""Bootstrap AMG processes that generate test functions with low Helmholtz residuals on a periodic domain."""
import logging
from typing import Tuple

import numpy as np
import scipy.sparse

import helmholtz as hm
from helmholtz.linalg import scaled_norm

_LOGGER = logging.getLogger(__name__)


def generate_test_matrix(a: scipy.sparse.spmatrix, num_growth_steps: int, growth_factor: int = 2,
                         num_bootstrap_steps: int = 1, aggregate_size: int = 4, num_sweeps: int = 10,
                         num_examples: int = None, print_frequency: int = None, initial_max_levels: int = 2) -> \
        Tuple[np.ndarray, hm.multilevel.Multilevel]:
    """
    Creates low-residual test functions and multilevel hierarchy on a large domain from the operator on a small window
    (the coarsest domain). This is similar to a full multigrid algorithm.

    Args:
        a: the operator on the coarsest domain.
        num_growth_steps: number of steps to increase the domain.
        growth_factor: by how much to increase the domain size at each growth step.
        num_bootstrap_steps: number of bootstrap steps to perform on each domain.
        aggregate_size: aggregate size = #fine vars per aggregate
        num_sweeps: number of relaxations or cycles to run on fine-level vectors to improve them.
        num_examples: number of test functions to generate. If None, uses 4 * np.prod(window_shape).
        print_frequency: print debugging convergence statements per this number of relaxation cycles/sweeps.
          None means no printouts.

    Returns:
        x: test matrix on the largest (final) domain.
        multilevel: multilevel hierarchy on the largest (final) domain.
    """
    # Initialize test functions (to random) and hierarchy at coarsest level.
    level = hm.multilevel.Level.create_finest_level(a)
    multilevel = hm.multilevel.Multilevel(level)
    # TODO(orenlivne): generalize to d-dimensions. This is specific to 1D.
    domain_shape = (a.shape[0],)
    x = hm.run.random_test_matrix(domain_shape, num_examples=num_examples)
    # Bootstrap at the current level.
    max_levels = initial_max_levels
    _LOGGER.info("Smallest domain size {}, bootstrap with {} levels".format(x.shape[0], max_levels))
    for i in range(num_bootstrap_steps):
        _LOGGER.info("Bootstrap step {}/{}".format(i + 1, num_bootstrap_steps))
        x, multilevel = bootstap(
            x, multilevel, max_levels, aggregate_size=aggregate_size, num_sweeps=num_sweeps, num_examples=num_examples,
            print_frequency=print_frequency)

    for l in range(num_growth_steps):
        _LOGGER.info("Growing domain {}/{} to size {}, max_levels {}".format(
            l + 1, num_growth_steps, growth_factor * x.shape[0], max_levels))
        # Tile solution and hierarchy on the twice larger domain.
        x = hm.linalg.tile_matrix(x, growth_factor)
        multilevel = multilevel.tile_csr_matrix(growth_factor)
        # Bootstrap at the current level.
        for i in range(num_bootstrap_steps):
            _LOGGER.info("Bootstrap step {}/{}".format(i + 1, num_bootstrap_steps))
            x, multilevel = bootstap(
                x, multilevel, max_levels, aggregate_size=aggregate_size, num_sweeps=num_sweeps,
                num_examples=num_examples, print_frequency=print_frequency)
        max_levels += 1

    return x, multilevel


def bootstap(x, multilevel: hm.multilevel.Multilevel, max_levels: int, aggregate_size: int = 4, num_sweeps: int = 10,
             threshold: float = 0.1, caliber: int = 2, interpolation_method: str = "svd",
             num_examples: int = None, print_frequency: int = None) -> \
        Tuple[np.ndarray, hm.multilevel.Multilevel]:
    """
    Improves test functions and a multilevel hierarchy on a fixed-size domain by bootstrapping.
    Args:
        x: test matrix.
        multilevel: multilevel hierarchy.
        aggregate_size: aggregate size = #fine vars per aggregate.
        num_sweeps: number of relaxations or cycles to run on fine-level vectors to improve them.
        threshold: relative reconstruction error threshold. Determines nc.
        caliber: interpolation caliber.
        interpolation_method: type of interpolation ("svd"|"ls").
        num_examples: number of test functions to generate. If None, uses 4 * np.prod(window_shape).

    Returns:
        improved x, multilevel hierarchy with the same number of levels.
    """
    # At the finest level, use the current multilevel hierarchy to run 'num_sweeps' cycles to improve x.
    finest = 0
    level = multilevel.level[finest]
    b = np.zeros_like(x)
    # TODO(orenlivne): update parameters of relaxation cycle to reasonable values if needed.
    if len(multilevel) == 1:
        def relax_cycle(x):
            return multilevel.relax_cycle(x, None, None, 1)
    else:
        def relax_cycle(x):
            return multilevel.relax_cycle(x, 2, 2, 30)
    _LOGGER.info("{} at level {}".format("Relax" if len(multilevel) == 1 else "Cycle", finest))
    x, _ = hm.run.relax_test_matrix(level.operator, level.rq, relax_cycle, x, num_sweeps)
    _LOGGER.info("lambda {}".format(multilevel.level[0].global_params.lam))

    # Recreate all coarse levels. One down-pass, relaxing at each level, hopefully starting from improved x so the
    # process improves all levels.
    # TODO(orenlivne): add nested bootstrap cycles if needed.
    new_multilevel = hm.multilevel.Multilevel(level)
    # Keep the x's of coarser levels in x_level; keep 'x' pointing to the finest test matrix.
    x_level = x
    for l in range(1, max_levels):
        _LOGGER.info("Coarsening level {}->{}".format(l - 1, l))
        domain_size = level.a.shape[0]
        r, p = create_transfer_operators(x_level, domain_size,
                                         aggregate_size=aggregate_size, threshold=threshold, caliber=caliber,
                                         interpolation_method=interpolation_method)
        # 'level' now becomes the next coarser level and x_level the corresponding test matrix.
        level = hm.multilevel.Level.create_coarse_level(level.a, level.b, level.global_params, r, p)
        new_multilevel.level.append(level)
        x_level = level.restrict(x_level)
        b = np.zeros_like(x_level)
        _LOGGER.info("Relax at level {}".format(l))
        x_level, _ = hm.run.relax_test_matrix(level.operator, level.rq, lambda x: level.relax(x, b), x_level,
                                                     num_sweeps=num_sweeps, print_frequency=print_frequency)
        _LOGGER.info("lambda {}".format(multilevel.level[0].global_params.lam))

    return x, new_multilevel


def fmg(multilevel, nu_pre: int = 1, nu_post: int = 1, nu_coarsest: int = 10, num_cycles: int = 1,
        num_examples: int = 1, num_cycles_finest: int = 1, finest: int = 0):
    """Runs an num_cycles-FMG algorithm to get an initial guess at the finest level of the multilevel hierarchy.
    Runs 'num_cycles' at the finest level and 'num_cycles' at all other levels"""
    coarsest = len(multilevel) - 1

    # Coarsest level initial guess.
    level = multilevel.level[coarsest]
    x = hm.run.random_test_matrix((level.a.shape[0],), num_examples=num_examples)
    level.global_params.lam = 0

    for l in range(coarsest, finest, -1):
        level = multilevel.level[l]
        x0 = x[:, 0]
        r_norm = scaled_norm(level.operator(x0))
        _LOGGER.debug("FMG level {} init |r| {:.8e} lam {:.5f}".format(l, r_norm, level.global_params.lam))
        for _ in range(num_cycles):
            x = multilevel.relax_cycle(x, nu_pre, nu_post, nu_coarsest, finest_level_ind=l)

        x = level.interpolate(x)
        level = multilevel.level[l - 1]
        x0 = x[:, 0]
        r_norm = scaled_norm(level.operator(x0))
        _LOGGER.debug("FMG level {} cycles {} |r| {:.8e} lam {:.5f}".format(l, num_cycles, r_norm, level.global_params.lam))

    l = finest
    x0 = x[:, 0]
    r_norm = scaled_norm(level.operator(x0))
    _LOGGER.debug("FMG level {} init |r| {:.8e} lam {:.5f}".format(l, r_norm, level.global_params.lam))
    for _ in range(num_cycles_finest):
        x = multilevel.relax_cycle(x, nu_pre, nu_post, nu_coarsest, finest_level_ind=l)
    x0 = x[:, 0]
    r_norm = scaled_norm(level.operator(x0))
    _LOGGER.debug("FMG level {} cycles {} |r| {:.8e} lam {:.5f}".format(l, num_cycles_finest, r_norm, level.global_params.lam))
    return x


def create_transfer_operators(x, domain_size: int, aggregate_size: int, threshold: float = 0.1, caliber: int = 2,
                              interpolation_method: str = "svd") -> \
        Tuple[hm.restriction.Restrictor, hm.interpolation.Interpolator]:
    """
    Creates the next coarse level's R and P operators.
    Args:
        x: fine-level test matrix.
        domain_size: #gridpoints in fine level.
        aggregate_size: aggregate size = #fine vars per aggregate
        threshold: relative reconstruction error threshold. Determines nc.
        caliber: interpolation caliber.
        interpolation_method: type of interpolation ("svd"|"ls").

    Returns: R, P
    """
    # TODO(oren): generalize the domain to the d-dimensional case. For now assuming 1D only.
    assert domain_size % aggregate_size == 0, \
        "Aggregate shape must divide the domain shape in every dimension"
    # assert all(ni % ai == 0 for ni, ai in zip(domain_shape, aggregate_shape)), \
    #     "Aggregate shape must divide the domain shape in every dimension"
    num_aggregates = domain_size // aggregate_size

    x_aggregate_t = x[:aggregate_size].transpose()
    r, s = hm.restriction.create_restriction(x_aggregate_t, threshold)
    nc = r.asarray().shape[0]
    _LOGGER.debug("Singular vals {}, nc {} interpolation error {:.3f}".format(
        s, nc, (sum(s[nc:] ** 2) / sum(s ** 2)) ** 0.5))
    r_csr = r.tile(num_aggregates)
    xc = r_csr.dot(x)
    p = hm.interpolation.create_interpolation(interpolation_method,
                                              r.asarray(), x_aggregate_t, xc.transpose(), domain_size, nc, caliber)
    return r, p
