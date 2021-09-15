"""Bootstrap AMG processes that generate test functions with low Helmholtz residuals on a periodic domain.
Runs multigrid cycles on A*x=lambda*x. Test functions are eigenvectors."""
import logging
from typing import Tuple

import numpy as np
import scipy.sparse

import helmholtz as hm
from helmholtz.linalg import scaled_norm

_LOGGER = logging.getLogger(__name__)


def generate_test_matrix(a: scipy.sparse.spmatrix, num_growth_steps: int, growth_factor: int = 2,
                         num_bootstrap_steps: int = 1, aggregate_size: int = 4, num_sweeps: int = 10,
                         num_examples: int = None, print_frequency: int = None, initial_max_levels: int = 2,
                         interpolation_method: str = "svd",
                         threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray, hm.hierarchy.multilevel.Multilevel]:
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
        initial_max_levels: number of levels to employ at the smallest domain size.
        interpolation_method: type of interpolation ("svd"|"ls").

    Returns:
        x: test matrix on the largest (final) domain.
        multilevel: multilevel hierarchy on the largest (final) domain.
    """
    # Initialize test functions (to random) and hierarchy at coarsest level.
    level = hm.repetitive.hierarchy.create_finest_level(a)
    multilevel = hm.hierarchy.multilevel.Multilevel.create(level)
    # TODO(orenlivne): generalize to d-dimensions. This is specific to 1D.
    domain_shape = (a.shape[0],)
    x = hm.solve.run.random_test_matrix(domain_shape, num_examples=num_examples)
    lam = 0
    # Bootstrap at the current level.
    max_levels = initial_max_levels
    _LOGGER.info("Smallest domain size {}, bootstrap with {} levels".format(x.shape[0], max_levels))
    for i in range(num_bootstrap_steps):
        _LOGGER.info("Bootstrap step {}/{}".format(i + 1, num_bootstrap_steps))
        x, lam, multilevel = bootstap(
            x, lam, multilevel, max_levels, aggregate_size=aggregate_size, num_sweeps=num_sweeps,
            num_examples=num_examples, print_frequency=print_frequency, interpolation_method=interpolation_method,
            threshold=threshold)

    for l in range(num_growth_steps):
        _LOGGER.info("Growing domain {}/{} to size {}, max_levels {}".format(
            l + 1, num_growth_steps, growth_factor * x.shape[0], max_levels))
        # Tile solution and hierarchy on the twice larger domain.
        x = hm.linalg.tile_matrix(x, growth_factor)
        multilevel = multilevel.tile_csr_matrix(growth_factor)
        # Bootstrap at the current level.
        for i in range(num_bootstrap_steps):
            _LOGGER.info("Bootstrap step {}/{}".format(i + 1, num_bootstrap_steps))
            x, lam, multilevel = bootstap(
                x, lam, multilevel, max_levels, aggregate_size=aggregate_size, num_sweeps=num_sweeps,
                num_examples=num_examples, print_frequency=print_frequency, interpolation_method=interpolation_method,
            threshold=threshold)
        max_levels += 1

    return x, lam, multilevel


def bootstap(x, lam, multilevel: hm.hierarchy.multilevel.Multilevel, max_levels: int, aggregate_size: int = 4,
             num_sweeps: int = 10, threshold: float = 0.1, caliber: int = 2, interpolation_method: str = "svd",
             num_examples: int = None, print_frequency: int = None) -> \
        Tuple[np.ndarray, hm.hierarchy.multilevel.Multilevel]:
    """
    Improves test functions and a multilevel hierarchy on a fixed-size domain by bootstrapping.
    Args:
        x: test matrix.
        lam: corresponding eigenvalue array to x's columns.
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
    level = multilevel[finest]
    b = np.zeros_like(x)
    # TODO(orenlivne): update parameters of relaxation cycle to reasonable values if needed.
    if len(multilevel) == 1:
        def eigen_cycle(x, lam):
            return hm.setup_eigen.eigensolver.eigen_cycle(multilevel, 1.0, None, None, 5).run((x, lam))
    else:
        def eigen_cycle(x, lam):
            return hm.setup_eigen.eigensolver.eigen_cycle(multilevel, 1.0, 2, 2, 30).run((x, lam))
    _LOGGER.info("{} at level {}".format("Relax" if len(multilevel) == 1 else "Cycle", finest))
    x, lam, _ = hm.solve.run.run_iterative_eigen_method(level.operator, eigen_cycle, x, lam, num_sweeps)
    _LOGGER.info("lambda {}".format(lam))

    # Recreate all coarse levels. One down-pass, relaxing at each level, hopefully starting from improved x so the
    # process improves all levels.
    # TODO(orenlivne): add nested bootstrap cycles if needed.
    new_multilevel = hm.hierarchy.multilevel.Multilevel.create(level)
    # Keep the x's of coarser levels in x_level; keep 'x' pointing to the finest test matrix.
    x_level = x
    for l in range(1, max_levels):
        _LOGGER.info("Coarsening level {}->{}".format(l - 1, l))
        domain_size = level.a.shape[0]
        r, p, _ = create_transfer_operators(x_level, domain_size,
                                         aggregate_size=aggregate_size, threshold=threshold, caliber=caliber,
                                         interpolation_method=interpolation_method)
        # 'level' now becomes the next coarser level and x_level the corresponding test matrix.
        level = hm.repetitive.hierarchy.create_tiled_coarse_level(level.a, level.b, r, p)
        new_multilevel.add(level)
        x_level = level.coarsen(x_level)
        b = np.zeros_like(x_level)
        _LOGGER.info("Relax at level {}".format(l))
        x_level, lam, _ = hm.solve.run.run_iterative_eigen_method(
            level.operator, lambda x, lam: (level.relax(x, b, lam), lam), x_level, lam, num_sweeps=num_sweeps,
            print_frequency=print_frequency)
        _LOGGER.info("lambda {}".format(lam))

    return x, lam, new_multilevel


def fmg(multilevel, nu_pre: int = 1, nu_post: int = 1, nu_coarsest: int = 10, num_cycles: int = 1,
        cycle_index: float = 1.0,  num_examples: int = 1, num_cycles_finest: int = 1, finest: int = 0):
    """Runs an num_cycles-FMG algorithm to get an initial guess at the finest level of the multilevel hierarchy.
    Runs 'num_cycles' at the finest level and 'num_cycles' at all other levels"""
    coarsest = len(multilevel) - 1

    # Coarsest level initial guess.
    level = multilevel[coarsest]
    x = hm.solve.run.random_test_matrix((level.a.shape[0],), num_examples=num_examples)
    lam = 0

    processor = hm.setup_eigen.eigensolver.EigenProcessor(multilevel, nu_pre, nu_post, nu_coarsest)
    for l in range(coarsest, finest, -1):
        level = multilevel[l]
        x0 = x[:, 0]
        r_norm = scaled_norm(level.operator(x0, lam))
        _LOGGER.debug("FMG level {} init |r| {:.8e} lam {:.5f}".format(l, r_norm, lam))
        eigen_cycle = hm.hierarchy.cycle.Cycle(processor, cycle_index, coarsest - l + 1, finest=l)
        for _ in range(num_cycles):
            x, lam = eigen_cycle.run((x, lam))

        x = level.interpolate(x)
        level = multilevel[l - 1]
        x0 = x[:, 0]
        r_norm = scaled_norm(level.operator(x0, lam))
        _LOGGER.debug("FMG level {} cycles {} |r| {:.8e} lam {:.5f}".format(l, num_cycles, r_norm, lam))

    l = finest
    x0 = x[:, 0]
    r_norm = scaled_norm(level.operator(x0, lam))
    _LOGGER.debug("FMG level {} init |r| {:.8e} lam {:.5f}".format(l, r_norm, lam))
    eigen_cycle = hm.setup_eigen.eigensolver.eigen_cycle(multilevel, cycle_index, nu_pre, nu_post, nu_coarsest, finest=l)
    for _ in range(num_cycles_finest):
        x = eigen_cycle.run()
    x0 = x[:, 0]
    r_norm = scaled_norm(level.operator(x0, lam))
    _LOGGER.debug("FMG level {} cycles {} |r| {:.8e} lam {:.5f}".format(l, num_cycles_finest, r_norm, lam))
    return x


def create_transfer_operators(x, domain_size: int, aggregate_size: int, threshold: float = 0.1, caliber: int = 2,
                              interpolation_method: str = "svd", max_coarsening_ratio: float = 0.5) -> \
        Tuple[hm.setup.coarsening.Coarsener, hm.setup.interpolation.Interpolator]:
    """
    Creates the next coarse level's R and P operators.
    Args:
        x: fine-level test matrix.
        domain_size: #gridpoints in fine level.
        aggregate_size: aggregate size = #fine vars per aggregate
        threshold: relative reconstruction error threshold. Determines nc.
        caliber: interpolation caliber.
        interpolation_method: type of interpolation ("svd"|"ls").
        max_coarsening_ratio: maximum allowed coarsening ratio. If exceeded at a certain aggregate size, we double
            it until it is reached (or when the aggregate size becomes too large, in which case an exception is raised).

    Returns: R, P, s = singular values array.
    """
    # TODO(oren): generalize the domain to the d-dimensional case. For now assuming 1D only.
    assert domain_size % aggregate_size == 0, \
        "Aggregate shape must divide the domain shape in every dimension"
    # assert all(ni % ai == 0 for ni, ai in zip(domain_shape, aggregate_shape)), \
    #     "Aggregate shape must divide the domain shape in every dimension"

    aggregate_size = 1
    coarsening_ratio = 1
    num_test_functions = x.shape[1]
    # Increase aggregate size until we reach a small enough coarsening ratio.
    while (aggregate_size <= domain_size // 2) and (coarsening_ratio > max_coarsening_ratio):
        aggregate_size *= 2
        num_windows = max((4 * aggregate_size) // num_test_functions, 1)
        x_aggregate_t = np.concatenate(
            tuple(hm.linalg.get_window(x, offset, aggregate_size) for offset in range(num_windows)), axis=1).transpose()
        r, s = hm.setup.coarsening.create_coarsening(x_aggregate_t, threshold)
        nc = r.asarray().shape[0]
        coarsening_ratio = nc / aggregate_size
        _LOGGER.debug("SVD {:2d} x {:2d} aggregate size {} nc {} cr {:.2f} interpolation error {:.3f} Singular vals {}".format(
            x_aggregate_t.shape[0], x_aggregate_t.shape[1], aggregate_size, nc, coarsening_ratio,
            (sum(s[nc:] ** 2) / sum(s ** 2)) ** 0.5, np.array2string(s, separator=", ", precision=2)))
    if (aggregate_size > domain_size // 2) or (coarsening_ratio > max_coarsening_ratio):
        raise Exception("Could not find a good coarsening ratio")

    num_aggregates = domain_size // aggregate_size
    r_csr = r.tile(num_aggregates)
    xc = r_csr.dot(x)

    # Create windows of fine-level test functions on disjoint aggregates (e.g., points 0..3, 4..7, etc. for
    # aggregate_size = 4). No need to use all windows of the domain in the least-squares fit. It's sufficient to use
    # >> than the interpolation caliber. Using 12 because we need (train, val, test) for LS fit, each of which is
    # say 4 times the caliber to be safe.
    num_windows = max(np.minimum(num_aggregates, (12 * caliber) // num_test_functions), 1)
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
    xc_disjoint_aggregate_t = np.concatenate(tuple(hm.linalg.get_window(xc, offset, num_coarse_vars)
                                                 for offset in range(num_windows)), axis=1).transpose()

    p = hm.setup.interpolation.create_interpolation_repetitive(
        interpolation_method, r.asarray(), x_disjoint_aggregate_t, xc_disjoint_aggregate_t, domain_size, caliber)
    return r, p, s
