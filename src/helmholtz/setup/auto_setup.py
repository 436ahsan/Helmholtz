"""Bootstrap AMG setup phase: an algorithm that generate test functions with low residuals and a multilevel hierarchy
to solve a linear system A*x=b. Test functions are NOT eigenvectors."""
import logging
import numpy as np
import scipy.sparse
from typing import Tuple
from scipy.linalg import norm

import helmholtz as hm
import helmholtz.setup.hierarchy as hierarchy

_LOGGER = logging.getLogger(__name__)


def setup(a: scipy.sparse.spmatrix,
          max_levels: int = 100000,
          num_bootstrap_steps: int = 1,
          num_sweeps: int = 10,
          num_examples: int = 20,
          num_test_examples: int = 5,
          print_frequency: int = None,
          interpolation_method: str = "ls",
          threshold: float = 0.1,
          max_coarsest_level_size: int = 10,
          x: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, hm.hierarchy.multilevel.Multilevel]:
    """
    Creates low-residual test functions and multilevel hierarchy for solving A*x=b.
    Args:
        a: left-hand-side matrix.
        max_levels: maximum number of levels in the hierarchy.
        num_bootstrap_steps:  number of bootstrap steps to perform at each level.
        num_sweeps: number of relaxations or cycles to run on fine-level vectors to improve them.
        num_examples: total number of test functions to generate.
        num_test_examples: number of test functions dedicated to testing (do not participate in SVD, LS fit).
        print_frequency: print debugging convergence statements per this number of relaxation cycles/sweeps.
          None means no printouts.
        interpolation_method: type of interpolation ("svd"|"ls").
        threshold: SVD coarsening accuracy threshold.
        max_coarsest_level_size: stop coarsening when we reach a coarse level with size <= max_coarsest_level_size
            (unless max_levels has been reached first).
        x: optional test function initial guess. If None, we start with random[-1,1]/

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
    if x is None:
        x = hm.solve.run.random_test_matrix(domain_shape, num_examples=num_examples)
    # Improve vectors with 1-level relaxation.
    _LOGGER.info("Relax at level {} size {}".format(finest, level.size))
    b = np.zeros_like(x)
    x, relax_conv_factor = hm.solve.run.run_iterative_method(
        level.operator, lambda x: level.relax(x, b), x,num_sweeps=num_sweeps)
    _LOGGER.info("Relax convergence factor {:.3f}".format(relax_conv_factor))
    _LOGGER.info("RER {:.3f}".format(norm(a.dot(x)) / norm(x)))

    # Bootstrap with an increasingly deeper hierarchy (add one level at a time).
    for num_levels in range(2, max_levels + 1):
        _LOGGER.info("-" * 80)
        _LOGGER.info("bootstrap at grid size {} with {} levels".format(x.shape[0], num_levels))
        for i in range(num_bootstrap_steps):
            _LOGGER.info("Bootstrap step {}/{}".format(i + 1, num_bootstrap_steps))
            x, multilevel = bootstap(x, multilevel, num_levels, relax_conv_factor,
                                     num_sweeps=num_sweeps, print_frequency=print_frequency,
                                     interpolation_method=interpolation_method, threshold=threshold,
                                     num_test_examples=num_test_examples)
            _LOGGER.info("RER {:.3f}".format(norm(a.dot(x)) / norm(x)))
        if multilevel.level[-1].size <= max_coarsest_level_size:
            break
    return x, multilevel


def bootstap(x, multilevel: hm.hierarchy.multilevel.Multilevel, num_levels: int, relax_conv_factor: float,
             num_sweeps: int = 10,
             threshold: float = 0.1,
             interpolation_method: str = "ls",
             num_test_examples: int = 5,
             print_frequency: int = None,
             fixed_aggregate_size: int = None) -> Tuple[np.ndarray, hm.hierarchy.multilevel.Multilevel]:
    """
    Improves test functions and a multilevel hierarchy on a fixed-size domain by bootstrapping.
    Args:
        x: test matrix.
        multilevel: initial multilevel hierarchy.
        num_levels: number of levels in the returned multilevel hierarchy.
        relax_conv_factor: relaxation convergence factor over the first 'num_sweeps', starting from a random
            initial guess.
        num_sweeps: number of relaxations or cycles to run on fine-level vectors to improve them.
        threshold: relative reconstruction error threshold. Determines nc.
        interpolation_method: type of interpolation ("svd"|"ls").
        num_test_examples: number of test functions dedicated to testing (do not participate in SVD, LS fit).
        print_frequency: print debugging convergence statements per this number of relaxation cycles/sweeps.
          None means no printouts.
        fixed_aggregate_size: if not None, forces this aggregate size throughout the domain.

    Returns:
        improved x, multilevel hierarchy with the same number of levels.
    """
    # At the finest level, use the current multilevel hierarchy to run relaxation cycles on A*x=0 to improve x.
    # TODO(orenlivne): update parameters of relaxation cycle to reasonable values if needed.
    def relax_cycle(x):
        return hm.solve.relax_cycle.relax_cycle(multilevel, 1.0, 2, 2, 4).run(x)
    level = multilevel.level[0]

    # First, test relaxation cycle convergence on a random vector. If slow, this yields a yet unseen slow to converge
    # error to add to the test function set, and indicate that we should NOT attempt to improve the current TFs with
    # relaxation cycle, since it will do more harm than good.
    y, conv_factor = hm.solve.run.run_iterative_method(level.operator, relax_cycle,
                                                       np.random.random((level.a.shape[0], 1)),
                                                       num_sweeps=num_sweeps, print_frequency=print_frequency)
    y = y.flatten()
    coarse_level = multilevel.level[1] if len(multilevel.level) > 1 else None
    _LOGGER.info("Relax cycle conv factor {:.3f} asymptotic RQ {:.3f} RER {:.3f} P error {:.3f}".format(
        conv_factor, level.rq(y), norm(level.operator(y)) / norm(y),
        norm(y - coarse_level.p.dot(coarse_level.r.dot(y))) / norm(y) if coarse_level is not None else -1))

    if conv_factor < relax_conv_factor:
        # Relaxation cycle is more efficient than relaxation, smooth the previous vectors.
        _LOGGER.info("Improving vectors by relaxation cycles")
        x, _ = hm.solve.run.run_iterative_method(level.operator, relax_cycle, x, num_sweeps)
    else:
        # Prepend the vector y to x, since our folds are sorted as fit, then test.
        x = np.concatenate((y[:, None], x), axis=1)
        _LOGGER.info("Added vector to TF set, #examples {}".format(x.shape[1]))

    # Recreate all coarse levels. One down-pass, relaxing at each level, hopefully starting from improved x so the
    # process improves all levels.
    # TODO(orenlivne): add nested bootstrap cycles if needed.
    new_multilevel = hm.hierarchy.multilevel.Multilevel(level)
    # Keep the x's of coarser levels in x_level; keep 'x' pointing to the finest test matrix.
    x_level = x
    for l in range(1, num_levels):
        _LOGGER.info("Coarsening level {}->{}".format(l - 1, l))
        x_fit, x_test = x_level[:, :-num_test_examples], x_level[:, -num_test_examples:]

        # Create the coarsening operator R.
        r, aggregates, nc, energy_error = \
            hm.setup.coarsening.create_coarsening_domain(x_fit, threshold=threshold,
                                                         fixed_aggregate_size=fixed_aggregate_size)
        _LOGGER.info("Agg {}".format(np.array([len(aggregate) for aggregate in aggregates])))
        _LOGGER.info("nc  {}".format(nc))
        _LOGGER.info("Energy error mean {:.4f} max {:.4f}".format(np.mean(energy_error), np.max(energy_error)))
        # _LOGGER.info("Aggregate {}".format(aggregates))
        mock_conv_factor = np.array(
            [hm.setup.auto_setup.mock_cycle_conv_factor(level, r, nu) for nu in np.arange(1, 12, dtype=int)])
        _LOGGER.info("Mock cycle conv factor {}".format(np.array2string(mock_conv_factor, precision=3)))

        # Create the interpolation operator P.
        p = create_interpolation(x_level, level.a, r, interpolation_method)
        for title, x_set in (("fit", x_fit), ("test", x_test)):
            error = norm(x_set - p.dot(r.dot(x_set)), axis=0) / norm(x_set, axis=0)
            error_a = norm(level.a.dot(x_set - p.dot(r.dot(x_set))), axis=0) / norm(x_set, axis=0)
            _LOGGER.info(
                "{:<4s} set size {:<2d} P L2 error mean {:.2f} max {:.2f} A error mean {:.2f} max {:.2f}".format(
                    title, len(error), np.mean(error), np.max(error), np.mean(error_a), np.max(error_a)))

        # 'level' now becomes the next coarser level and x_level the corresponding test matrix.
        level = hierarchy.create_coarse_level(level.a, level.b, r, p)
        _LOGGER.info("Level {} size {}".format(l, level.size))
        new_multilevel.level.append(level)
        if l < num_levels - 1:
            x_level = level.coarsen(x_level)
            b = np.zeros_like(x_level)
            _LOGGER.info("Relax at level {}".format(l))
            x_level, _ = hm.solve.run.run_iterative_method(
                level.operator, lambda x: level.relax(x, b), x_level,
                num_sweeps=num_sweeps, print_frequency=print_frequency)
    return x, new_multilevel


def mock_cycle_conv_factor(level, r, num_relax_sweeps):
    return hm.solve.run.run_iterative_method(
        level.operator,
        hm.solve.mock_cycle.MockCycle(lambda x, b: level.relax(x, b), r, num_relax_sweeps),
        hm.solve.run.random_test_matrix((level.size,), num_examples=1),
        num_sweeps=10)[1]


def create_interpolation(x: np.ndarray, a: scipy.sparse.csr_matrix, r: scipy.sparse.csr_matrix, method: str) -> \
    scipy.sparse.csr_matrix:
    if method == "svd":
        p = r.transpose()
    elif method == "ls":
        p, fit_error, val_error, test_error, alpha_opt = \
            hm.setup.interpolation.create_interpolation_least_squares_auto_nbhrs(x, a, r)
        # _LOGGER.info("fit error {}".format(fit_error))
        # _LOGGER.info("val error {}".format(val_error))
        # _LOGGER.info("test error {}".format(test_error))
        _LOGGER.info("P max error: fit {:.3f} val {:.3f} test {:.3f}; alpha mean {:.3f}".format(
            max(fit_error), max(val_error), max(test_error), alpha_opt.mean()
        ))
    else:
        raise Exception("Unsupported interpolation method '{}'".format(method))
    return p
