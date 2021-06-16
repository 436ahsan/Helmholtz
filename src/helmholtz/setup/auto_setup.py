"""Bootstrap AMG setup phase: an algorithm that generate test functions with low residuals and a multilevel hierarchy
to solve a linear system A*x=b. Test functions are NOT eigenvectors."""
import logging
import numpy as np
import scipy.sparse
from typing import Tuple

import helmholtz as hm
import helmholtz.setup.hierarchy as hierarchy

_LOGGER = logging.getLogger(__name__)


def setup(a: scipy.sparse.spmatrix,
          max_levels: int = 100000,
          num_bootstrap_steps: int = 1,
          num_sweeps: int = 10,
          num_examples: int = None, print_frequency: int = None,
          interpolation_method: str = "ls",
          threshold: float = 0.1,
          max_coarsest_level_size: int = 10) -> Tuple[np.ndarray, np.ndarray, hm.hierarchy.multilevel.Multilevel]:
    """
    Creates low-residual test functions and multilevel hierarchy for solving A*x=b.
    Args:
        a: left-hand-side matrix.
        max_levels: maximum number of levels in the hierarchy.
        num_bootstrap_steps:  number of bootstrap steps to perform at each level.
        num_sweeps: number of relaxations or cycles to run on fine-level vectors to improve them.
        num_examples: number of test functions to generate. If None, uses 4 * np.prod(window_shape).
        print_frequency: print debugging convergence statements per this number of relaxation cycles/sweeps.
          None means no printouts.
        interpolation_method: type of interpolation ("svd"|"ls").
        threshold: SVD coarsening accuracy threshold.
        max_coarsest_level_size: stop coarsening when we reach a coarse level with size <= max_coarsest_level_size
            (unless max_levels has been reached first).

    Returns:
        x: test matrix on the largest (final) domain.
        multilevel: multilevel hierarchy on the largest (final) domain.
    """
    # Initialize hierarchy to 1-level and fine-level test functions to random.
    finest = 0
    level = hierarchy.create_finest_level(a)
    multilevel = hm.hierarchy.multilevel.Multilevel(level)
    # TODO(orenlivne): generalize to d-dimensions. This is specific to 1D.
    domain_shape = (a.shape[0],)
    x = hm.solve.run.random_test_matrix(domain_shape, num_examples=num_examples)
    # Improve vectors with 1-level relaxation.
    _LOGGER.info("Relax at level {}".format(finest))
    b = np.zeros_like(x)
    x, conv_factor = hm.solve.run.run_iterative_method(
        level.operator, lambda x: level.relax(x, b), x,num_sweeps=num_sweeps)
    _LOGGER.info("Relax convergence factor {:.3f}".format(conv_factor))

    # Bootstrap with an increasingly deeper hierarchy (add one level at a time).
    for num_levels in range(2, max_levels + 1):
        _LOGGER.info("bootstrap with {} levels".format(x.shape[0], num_levels))
        for i in range(num_bootstrap_steps):
            _LOGGER.info("Bootstrap step {}/{}".format(i + 1, num_bootstrap_steps))
            x, multilevel = bootstap(x, multilevel, num_levels, num_sweeps=num_sweeps, print_frequency=print_frequency,
                                     interpolation_method=interpolation_method, threshold=threshold)
        if multilevel.level[-1].size <= max_coarsest_level_size:
            break
    return x, multilevel


def bootstap(x, multilevel: hm.hierarchy.multilevel.Multilevel, num_levels: int,
             num_sweeps: int = 10, threshold: float = 0.1, interpolation_method: str = "ls",
             print_frequency: int = None) -> \
        Tuple[np.ndarray, hm.hierarchy.multilevel.Multilevel]:
    """
    Improves test functions and a multilevel hierarchy on a fixed-size domain by bootstrapping.
    Args:
        x: test matrix.
        multilevel: initial multilevel hierarchy.
        num_levels: number of levels in the returned multilevel hierarchy.
        num_sweeps: number of relaxations or cycles to run on fine-level vectors to improve them.
        threshold: relative reconstruction error threshold. Determines nc.
        interpolation_method: type of interpolation ("svd"|"ls").
        print_frequency: print debugging convergence statements per this number of relaxation cycles/sweeps.
          None means no printouts.

    Returns:
        improved x, multilevel hierarchy with the same number of levels.
    """
    # At the finest level, use the current multilevel hierarchy to run relaxation cycles on A*x=0 to improve x.
    # TODO(orenlivne): update parameters of relaxation cycle to reasonable values if needed.
    def relax_cycle(x):
        return hm.solve.relax_cycle.relax_cycle(multilevel, 1.0, 2, 2, 4).run(x)
    level = multilevel.level[0]
    x, _ = hm.solve.run.run_iterative_method(level.operator, relax_cycle, x, num_sweeps)

    # Recreate all coarse levels. One down-pass, relaxing at each level, hopefully starting from improved x so the
    # process improves all levels.
    # TODO(orenlivne): add nested bootstrap cycles if needed.
    new_multilevel = hm.hierarchy.multilevel.Multilevel(level)
    # Keep the x's of coarser levels in x_level; keep 'x' pointing to the finest test matrix.
    x_level = x
    for l in range(1, num_levels):
        _LOGGER.info("Coarsening level {}->{}".format(l - 1, l))
        r, aggregates = hm.setup.coarsening.create_coarsening_full_domain(x_level, threshold=threshold)
        p = _create_interpolation(x_level, level.a, r, interpolation_method)

        # 'level' now becomes the next coarser level and x_level the corresponding test matrix.
        level = hierarchy.create_coarse_level(level.a, level.b, r, p)
        new_multilevel.level.append(level)
        if l < num_levels - 1:
            x_level = level.restrict(x_level)
            b = np.zeros_like(x_level)
            _LOGGER.info("Relax at level {}".format(l))
            x_level, _ = hm.solve.run.run_iterative_method(
                level.operator, lambda x: level.relax(x, b), x_level,
                num_sweeps=num_sweeps, print_frequency=print_frequency)
    return x, new_multilevel


def _create_interpolation(x: np.ndarray, a: scipy.sparse.csr_matrix, r: scipy.sparse.csr_matrix, method: str) -> \
    scipy.sparse.csr_matrix:
    if method == "svd":
        p = r.transpose()
    elif method == "ls":
        p, fit_error, val_error, test_error, alpha_opt = \
            hm.setup.interpolation.create_interpolation_least_squares_auto_nbhrs(x, a, r)
        _LOGGER.info("P error: fit {:.3f} val {:.3f} test {:.3f} alpha mean {:.3f}".format(
            max(fit_error), max(val_error), max(test_error), alpha_opt.mean()
        ))
    else:
        raise Exception("Unsupported interpolation method '{}'".format(method))
    return p
