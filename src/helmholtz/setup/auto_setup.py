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
          fine_location: np.ndarray,
          domain_size: float,
          max_levels: int = 100000,
          num_bootstrap_steps: int = 1,
          interpolation_method: str = "ls",
          neighborhood: str = "extended",
          repetitive: bool = False,
          max_coarsest_relax_conv_factor: float = 0.8,
          max_coarsest_level_size: int = 10,
          caliber: int = None,
          target_error: float = 0.2,
          leeway_factor: float = 1.2,
          symmetrize: bool = False) -> hierarchy.multilevel.Multilevel:
    """
    Creates low-residual test functions and multilevel hierarchy for solving A*x=b.
    Args:
        a: left-hand-side matrix.
        max_levels: maximum number of levels in the hierarchy.
        num_bootstrap_steps:  number of bootstrap steps to perform at each level.
        interpolation_method: type of interpolation ("svd"|"ls").
        neighborhood: "aggregate"|"extended" coarse neighborhood to interpolate from: only coarse variables in the
            aggregate, or the R*A*R^T sparsity pattern.
        repetitive: whether to exploit problem repetitiveness by creating a constant R stencil on all aggregates
            using windows from a single (or few) test vectors.
        max_coarsest_relax_conv_factor: if relaxation asymptotic convergence factor <= this value, stop coarsening.
        max_coarsest_level_size: stop coarsening when we reach a coarse level with size <= max_coarsest_level_size
            (unless max_levels has been reached first).
        leeway_factor: efficiency inflation factor past the point of diminishing returns to use in estimating where
            slowness starts.

    Returns:
        x: test matrix on the largest (final) domain.
        multilevel: multilevel hierarchy on the largest (final) domain.
    """
    # Initialize hierarchy to 1-level and fine-level test functions to random.
    min_relax_reduction_factor = 0.5
    min_relax_sweeps = 2
    level = hierarchy.create_finest_level(a)
    # Manually/separately set locations on the level object, as this is geometry-specific and Level is morre general.
    level.location = fine_location
    level.domain_size = domain_size
    multilevel = hm.hierarchy.multilevel.Multilevel.create(level)
    num_levels = 1
    shrinkage_factor, num_sweeps, residual, _, _, relax_conv_factor = \
        check_relaxation_speed(num_levels - 1, level, leeway_factor=leeway_factor)
    # Use at least nu sweeps such that shrinkage^nu <= 0.5, just to be safe.
    num_sweeps = max((num_sweeps, min_relax_sweeps,
                      int(np.ceil(np.log(min_relax_reduction_factor) / np.log(shrinkage_factor)))))
    _LOGGER.info("Using {} relaxations at level {} shrinkage {:.2f}".format(
        num_sweeps, num_levels - 1, shrinkage_factor))
    #_LOGGER.debug("residual {}".format(np.array2string(residual, separator=", ", precision=2)))

    # Create the hierarchy of coarser levels. For now, we use two-level bootstrap only.
    # Eventually, bootstrap may be needed with an increasingly deeper hierarchy (add one level at a time).
    for num_levels in range(2, max_levels + 1):
        if relax_conv_factor <= max_coarsest_relax_conv_factor or multilevel[-1].size <= max_coarsest_level_size:
            break
        _LOGGER.info("=" * 80)
        _LOGGER.info("Coarsening level {}->{}".format(num_levels - 2, num_levels - 1))
        level = build_coarse_level(level, num_sweeps, num_bootstrap_steps=num_bootstrap_steps,
                                   interpolation_method=interpolation_method, repetitive=repetitive,
                                   neighborhood=neighborhood, target_error=target_error, symmetrize=symmetrize,
                                   caliber=caliber)
        multilevel.add(level)
        shrinkage_factor, num_sweeps, residual, _, _, relax_conv_factor = \
            check_relaxation_speed(num_levels - 1, level, leeway_factor=leeway_factor)
        # Use at least nu sweeps such that shrinkage^nu <= 0.5, just to be safe.
        num_sweeps = max((num_sweeps, min_relax_sweeps,
                          int(np.ceil(np.log(min_relax_reduction_factor) / np.log(shrinkage_factor)))))
        _LOGGER.info("Using {} relaxations at level {}".format(num_sweeps, num_levels - 1))
        #_LOGGER.debug("residual {}".format(np.array2string(residual, separator=", ", precision=2)))
    return multilevel


def build_coarse_level(level: hm.hierarchy.multilevel.Level,
                       num_sweeps: int,
                       num_bootstrap_steps: int = 1,
                       max_caliber: int = 6,
                       caliber: int = None,
                       target_error: float = 0.2,
                       num_test_examples: int = 5,
                       interpolation_method: str = "ls",
                       neighborhood: str = "extended",
                       repetitive: bool = False,
                       symmetrize: bool = False) -> \
        Tuple[np.ndarray, np.ndarray, hm.hierarchy.multilevel.Multilevel]:
    """
    Creates the next-coarser level in the multilevel hierarchy.
    Args:
        level: finest level to coarsen.
        num_sweeps: number of relaxation sweeps to perform on test vectors.
        num_bootstrap_steps:  number of bootstrap steps to perform at each level.
        num_test_examples: number of test functions dedicated to testing (do not participate in SVD, LS fit).
        interpolation_method: type of interpolation ("svd"|"ls").
        neighborhood: "aggregate"|"extended" coarse neighborhood to interpolate from: only coarse variables in the
            aggregate, or the R*A*R^T sparsity pattern.
        repetitive: whether to exploit problem repetitiveness by creating a constant R stencil on all aggregates
            using windows from a single (or few) test vectors.
        max_caliber: estimated maximum interpolation caliber. Used to sample enough windows.

    Returns:
        x: test matrix on the largest (final) domain.
        multilevel: multilevel hierarchy on the largest (final) domain.
    """
    multilevel = hm.hierarchy.multilevel.Multilevel.create(level)
    a = level.a

    # Initialize fine-level test functions to random. Use enough to have enough windows if repetitive.
    num_examples = 20 if not repetitive else int(np.ceil(24 * max_caliber / level.size))
    x = hm.solve.run.random_test_matrix((a.shape[0],), num_examples=num_examples)

    # Improve vectors with 1-level relaxation.
    _LOGGER.info("Generating {} TV with {} sweeps".format(num_examples, num_sweeps))
    b = np.zeros_like(x)
    x = hm.solve.run.run_iterative_method(
        level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)[0]
    _LOGGER.info("RER {:.3f}".format(norm(a.dot(x)) / norm(x)))

    # Bootstrap to improve vectors, R, P.
    num_levels = 2
    _LOGGER.info("bootstrap on grid size {} with {} levels".format(x.shape[0], num_levels))
    _LOGGER.info("-" * 80)
    for i in range(num_bootstrap_steps):
        _LOGGER.info("Bootstrap step {}/{}".format(i + 1, num_bootstrap_steps))
        # Set relax_conv_factor to a high value so that we never append a bootstrap vector to the TV set.
        x, multilevel = hm.setup.auto_setup.bootstrap(
            x, multilevel, num_levels,
            num_sweeps=num_sweeps, interpolation_method=interpolation_method, neighborhood=neighborhood,
            repetitive=repetitive, num_test_examples=num_test_examples, max_caliber=max_caliber,
            target_error=target_error, symmetrize=symmetrize, caliber=caliber)
        _LOGGER.info("RER {:.6f}".format(norm(a.dot(x)) / norm(x)))
        _LOGGER.info("-" * 80)
    return multilevel[1]


def check_relaxation_speed(index, level, leeway_factor: float = 1.2):
    """
    Checks relaxation speed and shrinkage.

    Args:
        index:
        level:
        leeway_factor: efficiency inflation factor past the point of diminishing returns to use in estimating where
            slowness starts.

    Returns:
        relaxation shrinkage factor, num_sweeps at PODR, convergence factor
    """
    factor, num_sweeps, residual, conv, rer, relax_conv_factor = hm.solve.smoothing.shrinkage_factor(
        level.operator, lambda x, b: level.relax(x, b), (level.a.shape[0],),
        max_sweeps=20, slow_conv_factor=1.1, leeway_factor=leeway_factor)
    work = 1
    _LOGGER.info("level {} size {} relax conv {:.2f} shrinkage {:.2f} PODR RER {:.2f} after {} sweeps. "
                 "Work {:.1f} eff {:.2f}".format(
        index, level.size, relax_conv_factor, factor, np.mean(rer[num_sweeps]), num_sweeps, work,
        np.mean(residual[num_sweeps] / residual[0]) ** (1 / (num_sweeps * work))))
    return factor, num_sweeps, residual, conv, rer, relax_conv_factor


def bootstrap(x,
              multilevel: hm.hierarchy.multilevel.Multilevel,
              num_levels: int,
              num_sweeps: int = 10,
              interpolation_method: str = "ls",
              num_test_examples: int = 5,
              print_frequency: int = None,
              max_aggregate_size: int = 8,
              max_conv_factor: float = 0.4,
              neighborhood: str = "extended",
              max_caliber: int = 6,
              caliber: int = None,
              target_error: float = 0.2,
              repetitive: bool = False,
              aggregate_size: int = None,
              symmetrize: bool = False) -> Tuple[np.ndarray, hm.hierarchy.multilevel.Multilevel]:
    """
    Improves test functions and a multilevel hierarchy on a fixed-size domain by bootstrapping.
    Args:
        x: test matrix.
        multilevel: initial multilevel hierarchy.
        num_levels: number of levels in the returned multilevel hierarchy.
        num_sweeps: number of relaxations or cycles to run on fine-level vectors to improve them.
        interpolation_method: type of interpolation ("svd"|"ls").
        num_test_examples: number of test functions dedicated to testing (do not participate in SVD, LS fit).
        print_frequency: print debugging convergence statements per this number of relaxation cycles/sweeps.
          None means no printouts.
        max_aggregate_size: maximum allowed aggregate size.
        max_conv_factor: max convergence factor to allow. NOTE: in principle, should be derived from cycle index.
        neighborhood: "aggregate"|"extended" coarse neighborhood to interpolate from: only coarse variables in the
            aggregate, or the R*A*R^T sparsity pattern.
        repetitive: whether to exploit problem repetitiveness by creating a constant R stencil on all aggregates
            using windows from a single (or few) test vectors.
        symmetrize: whether to use the sparse symmetric coarse-level operator Q*A*P or P^T*A*P.

    Returns:
        improved x, multilevel hierarchy with the same number of levels.
    """
    # At the finest level, use the current multilevel hierarchy to run relaxation cycles on A*x=0 to improve x.
    # TODO(orenlivne): update parameters of relaxation cycle to reasonable values if needed.
    def relax_cycle(x):
        return hm.solve.relax_cycle.relax_cycle(multilevel, 1.0, 2, 2, 4).run(x)
    level = multilevel[0]

    # First, test relaxation cycle convergence on a random vector. If slow, this yields a yet unseen slow to converge
    # error to add to the test function set, and indicate that we should NOT attempt to improve the current TFs with
    # relaxation cycle, since it will do more harm than good.
    y, conv_factor = hm.solve.run.run_iterative_method(level.operator, relax_cycle,
                                                       np.random.random((level.a.shape[0], 1)),
                                                       num_sweeps=20, print_frequency=print_frequency)
    y = y.flatten()
    coarse_level = multilevel[1] if len(multilevel) > 1 else None
    _LOGGER.info("Relax cycle conv factor {:.3f} asymptotic RQ {:.3f} RER {:.3f} P error {:.3f}".format(
        conv_factor, level.rq(y), norm(level.operator(y)) / norm(y),
        norm(y - coarse_level.interpolate(coarse_level.coarsen(y))) / norm(y) if coarse_level is not None else -1))

    _LOGGER.info("Improving vectors by relaxation cycles")
    x, _ = hm.solve.run.run_iterative_method(level.operator, relax_cycle, x, num_sweeps)

    # Recreate all coarse levels. One down-pass, relaxing at each level, hopefully starting from improved x so the
    # process improves all levels.
    # TODO(orenlivne): add nested bootstrap cycles if needed.
    new_multilevel = hm.hierarchy.multilevel.Multilevel.create(level)
    # Keep the x's of coarser levels in x_level; keep 'x' pointing to the finest test matrix.
    x_level = x
    for l in range(1, num_levels):
        _LOGGER.info("Coarsening level {}->{}".format(l - 1, l))
        x_fit, x_test = x_level[:, :-num_test_examples], x_level[:, -num_test_examples:]

        # Create the coarsening operator R.
        coarsener = hm.setup.coarsening_uniform.UniformCoarsener(
            level, x, num_sweeps, repetitive=repetitive, max_aggregate_size=max_aggregate_size)
        info = coarsener.get_coarsening_info(1, fmt="dataframe")
        #_LOGGER.debug(info)
        r, aggregate_size, nc, cr, mean_energy_error, mock_conv, mock_work, mock_efficiency = \
            coarsener.get_optimal_coarsening(max_conv_factor, aggregate_size=aggregate_size)
        _LOGGER.info("R {} a {} nc {} cr {:.2f} mean_energy_error {:.4f}; mock cycle num_sweeps {} conv {:.2f} "
                     "eff {:.2f}".format(
            r.shape, aggregate_size, nc, cr, mean_energy_error, num_sweeps, mock_conv, mock_efficiency))

        # Calculate local Mock cycle rates.
        # TODO(oren): remove this double call to creating R. This is only a convenience so we can tile it on
        # a smaller sub-domain.
        coarsener, _ = hm.repetitive.locality.create_coarsening(x, aggregate_size, nc, normalize=False)
        m = 4
        subdomain_size = m * aggregate_size
        a_subdomain = level.a[:subdomain_size, :subdomain_size]
        level_subdomain = hm.setup.hierarchy.create_finest_level(a_subdomain)
        r_subdomain = coarsener.tile(m)
        nu_values = np.union1d(np.array(list(range(1, 4))), [num_sweeps])
        mock_conv_factor = np.array([hm.setup.auto_setup.mock_cycle_conv_factor(level_subdomain, r_subdomain, nu)
                                     for nu in nu_values])
        _LOGGER.info("Mock cycle conv factor {} for nu {}".format(
            np.array2string(mock_conv_factor, precision=3), np.array2string(nu_values)))

        # Create the interpolation operator P.
        p = create_interpolation(
            x_level, level.a, r, level.location, level.domain_size, interpolation_method, aggregate_size=aggregate_size, num_components=nc, max_caliber=max_caliber,
            neighborhood=neighborhood, repetitive=repetitive, target_error=target_error, caliber=caliber)
        coarse_location = hm.setup.geometry.coarse_locations(level.location, aggregate_size, nc)

        # 'level' now becomes the next coarser level and x_level the corresponding test matrix.
        if symmetrize:
            q = hm.repetitive.symmetry.symmetrize(r, level.a.dot(p), aggregate_size, nc)
        else:
            q = p.T

        # Measure two-level rates. This should be local, but for now using global.
        ml = hm.repetitive.locality.create_two_level_hierarchy_from_matrix(
            level.a, level.location, r, p, q, aggregate_size, nc, m=m)
        two_level_conv = np.array([hm.repetitive.locality.two_level_conv_factor(
            ml, nu, nu_coarsest=500, print_frequency=None)[1] for nu in nu_values])
        _LOGGER.info("2-level cycle conv factor {} for nu {}".format(
            np.array2string(two_level_conv, precision=3), np.array2string(nu_values)))

        coarse_level = hierarchy.create_coarse_level(level.a, level.b, r, p, q)
        coarse_level.location = coarse_location
        coarse_level.domain_size = level.domain_size
        level = coarse_level
        _LOGGER.info("Level {} size {}".format(l, level.size))
        new_multilevel.add(level)
        if l < num_levels - 1:
            x_level = level.coarsen(x_level)
            b = np.zeros_like(x_level)
            _LOGGER.info("Relax at level {}".format(l))
            x_level, _ = hm.solve.run.run_iterative_method(
                level.operator, lambda x: level.relax(x, b), x_level,
                num_sweeps=num_sweeps, print_frequency=print_frequency)
    return x, new_multilevel


def mock_cycle_conv_factor(level, r, num_relax_sweeps, print_frequency: int = None,
                           num_tf_sweeps: int = 15):
    return hm.solve.run.run_iterative_method(
        level.operator,
        hm.solve.mock_cycle.MockCycle(lambda x, b: level.relax(x, b), r, num_relax_sweeps),
        hm.solve.run.random_test_matrix((level.size,), num_examples=1),
        num_sweeps=num_tf_sweeps, print_frequency=print_frequency)[1]


def create_interpolation(x: np.ndarray,
                         a: scipy.sparse.csr_matrix,
                         r: scipy.sparse.csr_matrix,
                         fine_location: np.ndarray,
                         domain_size: float,
                         method: str,
                         aggregate_size: int = None, num_components: int = None,
                         neighborhood: str = "extended",
                         caliber: int = None,
                         fit_scheme: str = "ridge",
                         max_caliber: int = 6,
                         target_error: float = 0.2,
                         repetitive: bool = False,
                         weighted: bool = False,
                         num_windows: int = None) -> scipy.sparse.csr_matrix:
    """
    :param x:
    :param a:
    :param r:
    :param method:
    :param aggregate_size:
    :param num_components:
    :param neighborhood:
    :param caliber:
    :param fit_scheme: whether to use regularized unweighted LS ("ridge") or plain LS ("plain").
    :param max_caliber:
    :param target_error:
    :param repetitive:
    :param weighted: use weighting in LS or not.
    :return: interpolation matrix.
    """
    if method == "svd":
        p = r.T
    elif method == "ls":
        # TODO(oren): replace caliber by true max_caliber in this call (right now 'max_caliber' is interpreted here
        # as the fixed interpolation caliber returned; make the call loop over all calibers and return the desirable
        # one).
        p = hm.setup.interpolation.create_interpolation_least_squares_domain(
            x, a, r, fine_location, domain_size, aggregate_size=aggregate_size, num_components=num_components, neighborhood=neighborhood,
            repetitive=repetitive, caliber=caliber, max_caliber=max_caliber, target_error=target_error,
            fit_scheme=fit_scheme, weighted=weighted, num_windows=num_windows)
    else:
        raise Exception("Unsupported interpolation method '{}'".format(method))
    return p


def get_test_matrix(a, num_sweeps, num_examples: int = None):
    level = hm.setup.hierarchy.create_finest_level(a)
    x = hm.solve.run.random_test_matrix((a.shape[0],), num_examples=num_examples)
    b = np.zeros_like(x)
    x, _ = hm.solve.run.run_iterative_method(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)
    return x
