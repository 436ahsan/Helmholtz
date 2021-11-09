"""Bootstrap AMG processes that generate test functions with low Helmholtz residuals on a periodic domain.
Runs multigrid cycles on A*x=0. Test functions are NOT eigenvectors."""
import logging
from typing import Tuple

import numpy as np
import scipy.sparse

import helmholtz as hm
import helmholtz.repetitive.hierarchy_repetitive as hierarchy
from helmholtz.linalg import scaled_norm
from helmholtz.hierarchy.multilevel import Multilevel
from .interpolation_repetitive import Interpolator
from .coarsening_repetitive import Coarsener

_LOGGER = logging.getLogger(__name__)


def generate_test_matrix(a: scipy.sparse.spmatrix, num_growth_steps: int, growth_factor: int = 2,
                         num_bootstrap_steps: int = 1, num_sweeps: int = 10,
                         num_examples: int = None, print_frequency: int = None, initial_max_levels: int = 2,
                         interpolation_method: str = "svd",
                         threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray, Multilevel]:
    """
    Creates low-residual test functions and multilevel hierarchy on a large domain from the operator on a small window
    (the coarsest domain). This is similar to a full multigrid algorithm.

    Args:
        a: the operator on the coarsest domain.
        num_growth_steps: number of steps to increase the domain.
        growth_factor: by how much to increase the domain size at each growth step.
        num_bootstrap_steps: number of bootstrap steps to perform on each domain.
        num_sweeps: number of relaxations or cycles to run on fine-level vectors to improve them.
        num_examples: number of test functions to generate. If None, uses 4 * np.prod(window_shape).
        print_frequency: print debugging convergence statements per this number of relaxation cycles/sweeps.
          None means no printouts.
        initial_max_levels: number of levels to employ at the smallest domain size.
        interpolation_method: type of interpolation ("svd"|"ls").
        threshold: coarsening accuracy threshold.

    Returns:
        x: test matrix on the largest (final) domain.
        multilevel: multilevel hierarchy on the largest (final) domain.
    """
    # Initialize test functions (to random) and hierarchy at coarsest level.
    level = hierarchy.create_finest_level(a)
    multilevel = Multilevel.create(level)
    # TODO(orenlivne): generalize to d-dimensions. This is specific to 1D.
    domain_shape = (a.shape[0],)
    x = hm.solve.run.random_test_matrix(domain_shape, num_examples=num_examples)
    # Bootstrap at the current level.
    max_levels = initial_max_levels
    _LOGGER.info("Smallest domain size {}, bootstrap with {} levels".format(x.shape[0], max_levels))
    for i in range(num_bootstrap_steps):
        _LOGGER.info("Bootstrap step {}/{}".format(i + 1, num_bootstrap_steps))
        x, multilevel = bootstap(x, multilevel, max_levels, num_sweeps=num_sweeps, print_frequency=print_frequency,
                                 interpolation_method=interpolation_method, threshold=threshold)

    for l in range(num_growth_steps):
        _LOGGER.info("Growing domain {}/{} to size {}, max_levels {}".format(
            l + 1, num_growth_steps, growth_factor * x.shape[0], max_levels))
        # Tile solution and hierarchy on the twice larger domain.
        x = hm.linalg.tile_matrix(x, growth_factor)
        multilevel = multilevel.tile_csr_matrix(growth_factor)
        # Bootstrap at the current level.
        for i in range(num_bootstrap_steps):
            _LOGGER.info("Bootstrap step {}/{}".format(i + 1, num_bootstrap_steps))
            x, multilevel = bootstap(x, multilevel, max_levels, num_sweeps=num_sweeps, print_frequency=print_frequency,
                                     interpolation_method=interpolation_method, threshold=threshold)
        max_levels += 1

    return x, multilevel


def bootstap(x, multilevel: Multilevel, max_levels: int,
             num_sweeps: int = 10, threshold: float = 0.1, caliber: int = 2, interpolation_method: str = "svd",
             print_frequency: int = None) -> \
        Tuple[np.ndarray, Multilevel]:
    """
    Improves test functions and a multilevel hierarchy on a fixed-size domain by bootstrapping.
    Args:
        x: test matrix.
        multilevel: multilevel hierarchy.
        num_sweeps: number of relaxations or cycles to run on fine-level vectors to improve them.
        threshold: relative reconstruction error threshold. Determines nc.
        caliber: interpolation caliber.
        interpolation_method: type of interpolation ("svd"|"ls").

    Returns:
        improved x, multilevel hierarchy with the same number of levels.
    """
    # At the finest level, use the current multilevel hierarchy to run 'num_sweeps' cycles to improve x.
    finest = 0
    level = multilevel[finest]
    b = np.zeros_like(x)
    # TODO(orenlivne): update parameters of relaxation cycle to reasonable values if needed.
    if len(multilevel) == 1:
        def relax_cycle(x):
            return hm.solve.relax_cycle.relax_cycle(multilevel, 1.0, None, None, 1).run(x)
    else:
        def relax_cycle(x):
            return hm.solve.relax_cycle.relax_cycle(multilevel, 1.0, 2, 2, 4).run(x)
    _LOGGER.info("{} at level {}".format("Relax" if len(multilevel) == 1 else "Cycle", finest))
    x, _ = hm.solve.run.run_iterative_method(level.operator, relax_cycle, x, num_sweeps)

    # Recreate all coarse levels. One down-pass, relaxing at each level, hopefully starting from improved x so the
    # process improves all levels.
    # TODO(orenlivne): add nested bootstrap cycles if needed.
    new_multilevel = Multilevel.create(level)
    # Keep the x's of coarser levels in x_level; keep 'x' pointing to the finest test matrix.
    x_level = x
    for l in range(1, max_levels):
        _LOGGER.info("Coarsening level {}->{}".format(l - 1, l))
        domain_size = level.a.shape[0]
        residual_level = level.operator(x_level)
        r, p, _ = create_transfer_operators(x_level, residual_level, domain_size, threshold=threshold,
                                            caliber=caliber, interpolation_method=interpolation_method)
        # 'level' now becomes the next coarser level and x_level the corresponding test matrix.
        level = hierarchy.create_tiled_coarse_level(level.a, level.b, r, p)
        new_multilevel.add(level)
        if l < max_levels - 1:
            x_level = level.coarsen(x_level)
            b = np.zeros_like(x_level)
            _LOGGER.info("Relax at level {}".format(l))
            x_level, _ = hm.solve.run.run_iterative_method(level.operator, lambda x: level.relax(x, b), x_level,
                                                           num_sweeps=num_sweeps, print_frequency=print_frequency)

    return x, new_multilevel


def fmg(multilevel, nu_pre: int = 1, nu_post: int = 1, nu_coarsest: int = 10, num_cycles: int = 1,
        cycle_index: float = 1.0,  num_examples: int = 1, num_cycles_finest: int = 1, finest: int = 0):
    """Runs an num_cycles-FMG algorithm to get an initial guess at the finest level of the multilevel hierarchy.
    Runs 'num_cycles' at the finest level and 'num_cycles' at all other levels"""
    coarsest = len(multilevel) - 1

    # Coarsest level initial guess.
    level = multilevel[coarsest]
    x = hm.solve.run.random_test_matrix((level.a.shape[0],), num_examples=num_examples)

    processor = hm.solve.relax_cycle.RelaxCycleProcessor(multilevel, nu_pre, nu_post, nu_coarsest)
    for l in range(coarsest, finest, -1):
        level = multilevel[l]
        x0 = x[:, 0]
        r_norm = scaled_norm(level.operator(x0))
        x_norm = scaled_norm(x0)
        _LOGGER.debug("FMG level {} init |r| {:.8e} RER {:.5f}".format(l, r_norm, r_norm / x_norm))
        relax_cycle = hm.hierarchy.cycle.Cycle(processor, cycle_index, coarsest - l + 1, finest=l)
        for _ in range(num_cycles):
            x = relax_cycle.run((x))

        x = level.interpolate(x)
        level = multilevel[l - 1]
        x0 = x[:, 0]
        r_norm = scaled_norm(level.operator(x0))
        x_norm = scaled_norm(x0)
        _LOGGER.debug("FMG level {} cycles {} |r| {:.8e} RER {:.5f}".format(l, num_cycles, r_norm, r_norm / x_norm))

    l = finest
    x0 = x[:, 0]
    r_norm = scaled_norm(level.operator(x0))
    x_norm = scaled_norm(x0)
    _LOGGER.debug("FMG level {} init |r| {:.8e} RER {:.5f}".format(l, r_norm, r_norm / x_norm))
    relax_cycle = hm.solve.relax_cycle.relax_cycle(multilevel, cycle_index, nu_pre, nu_post, nu_coarsest, finest=l)
    for _ in range(num_cycles_finest):
        x = relax_cycle.run()
    x0 = x[:, 0]
    r_norm = scaled_norm(level.operator(x0))
    x_norm = scaled_norm(x0)
    _LOGGER.debug("FMG level {} cycles {} |r| {:.8e} RR {:.5f}".format(l, num_cycles_finest, r_norm, r_norm / x_norm))
    return x


def create_transfer_operators(x, residual: np.ndarray, domain_size: int, threshold: float = 0.1, caliber: int = 2,
                              interpolation_method: str = "svd", max_coarsening_ratio: float = 0.5) -> \
        Tuple[Coarsener,Interpolator]:
    """
    Creates the next coarse level's R and P operators.
    Args:
        x: fine-level test matrix.
        residual: fine-level residual matrix.
        domain_size: #gridpoints in fine level.
        threshold: relative reconstruction error threshold. Determines nc.
        caliber: interpolation caliber.
        interpolation_method: type of interpolation ("svd"|"ls").
        max_coarsening_ratio: maximum allowed coarsening ratio. If exceeded at a certain aggregate size, we double
            it until it is reached (or when the aggregate size becomes too large, in which case an exception is raised).

    Returns: R, P, s = singular value array.
    """
    aggregate_size = 1
    coarsening_ratio = 1
    num_test_functions = x.shape[1]
    # Increase aggregate size until we reach a small enough coarsening ratio.
    while (aggregate_size <= domain_size // 2) and (coarsening_ratio > max_coarsening_ratio):
        aggregate_size *= 2
        num_windows = max((4 * aggregate_size) // num_test_functions, 1)
        x_aggregate_t = np.concatenate(
            tuple(hm.linalg.get_window(x, offset, aggregate_size) for offset in range(num_windows)), axis=1).transpose()
        r, s = hm.repetitive.coarsening_repetitive.create_coarsening(x_aggregate_t, threshold)
        nc = r.asarray().shape[0]
        coarsening_ratio = nc / aggregate_size
        _LOGGER.debug("SVD {:2d} x {:2d} nc {} cr {:.2f} error {:.3f} Singular vals {}"
                      " error {}".format(
            x_aggregate_t.shape[0], x_aggregate_t.shape[1], nc, coarsening_ratio,
            (sum(s[nc:] ** 2) / sum(s ** 2)) ** 0.5, np.array2string(s, separator=", ", precision=2),
            np.array2string((1 - np.cumsum(s**2)/sum(s**2))**0.5, separator=", ", precision=2)))
    if (aggregate_size > domain_size // 2) or (coarsening_ratio > max_coarsening_ratio):
        raise Exception("Could not find a good coarsening ratio")

    num_aggregates = domain_size // aggregate_size
    r_csr = r.tile(num_aggregates)
    xc = r_csr.dot(x)

    x_disjoint_aggregate_t, xc_disjoint_aggregate_t = \
        hm.setup.sampling.get_disjoint_windows(x, xc, residual, aggregate_size, nc, caliber)

    p = hm.setup.interpolation.create_interpolation_repetitive(
        interpolation_method, r.asarray(), x_disjoint_aggregate_t, xc_disjoint_aggregate_t, domain_size, nc)
    return r, p, s


def create_repetitive_coarsening(x, num_windows, aggregate_size, threshold):
    """Returns a coarsening of a repetitive Helmholtz problem a from windows of the test matrix x."""
    return hm.repetitive.coarsening_repetitive.create_coarsening(
        np.concatenate(tuple(hm.linalg.get_window(x, offset, aggregate_size)
                                     for offset in range(num_windows)), axis=1).transpose(),
        threshold)
