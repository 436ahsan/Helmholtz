import helmholtz as hm
import helmholtz.repetitive.coarsening_repetitive as hrc
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import norm

_LOGGER = logging.getLogger(__name__)


def mock_cycle_rate(a, aggregate_size: int, num_components, nu_values: np.ndarray = np.arange(1, 12),
                    r: np.ndarray = None):
    # Create fine-level matrix.
    n = a.shape[0]
    level = hm.setup.hierarchy.create_finest_level(a)
    multilevel = hm.setup.hierarchy.multilevel.Multilevel.create(level)

    # 'location' is an array of variable locations at all levels. Used only for interpolation neighbor determination.
    # Finest-level variable locations are assumed to be [0..n-1], i.e. a domain of size n with meshsize h = 1.
    level.location = np.arange(n)
    # Relaxation shrinkage.
    method_info = hm.solve.smoothing.check_relax_cycle_shrinkage(
        multilevel, num_levels=1, leeway_factor=1.3, slow_conv_factor=0.95, num_examples=5,
        print_frequency=None, plot=False)
    info = method_info["relax"]
    shrinkage = info[0]
    num_sweeps = 2 * method_info["relax"][1]
    # Create relaxed TVs.
    x_random = hm.solve.run.random_test_matrix((level.a.shape[0],), num_examples=4)
    b = np.zeros_like(x_random)
    x = hm.solve.run.run_iterative_method(
        level.operator, lambda x: level.relax(x, b), x_random, num_sweeps=num_sweeps)[0]
    # Create coarsening.
    if r is None:
        r, s = hm.repetitive.locality.create_coarsening(x, aggregate_size, num_components, normalize=False)
        r = r.asarray()
    #    r = np.array([[0.5, -0.5]])
    #    print(kh, r)
    R = hrc.Coarsener(r).tile(level.a.shape[0] // aggregate_size)
    return [shrinkage, 2 * num_sweeps] + [hm.setup.auto_setup.mock_cycle_conv_factor(level, R, nu) for nu in nu_values]


def compare_coarsening(level,
                       coarsening_types,
                       nu,
                       domain_size: float,
                       aggregate_size: int,
                       num_components: int,
                       ideal_tv: bool = False,
                       num_examples: int = 5,
                       nu_values: np.ndarray = np.arange(1, 12),
                       interpolation_method: str = "ls",
                       fit_scheme: str = "ridge",
                       weighted: bool = False,
                       neighborhood: str = "extended",
                       repetitive: bool = False,
                       nu_coarsest: int = -1,
                       m: int = None,
                       print_frequency: int = None):
    # Generate initial test vectors.
    if m is None:
        m = level.size // aggregate_size
    subdomain_size = m * aggregate_size
    a_subdomain = level.a[:subdomain_size, :subdomain_size]
    # if m is None:
    #     m = level.size // aggregate_size
    # _LOGGER.info("Domain size {}".format(m * aggregate_size))
    if ideal_tv:
        _LOGGER.info("Generating {} ideal TVs".format(num_examples))
        x, lam = hm.analysis.ideal.ideal_tv(level.a, num_examples)
    else:
        _LOGGER.info("Generating {} TVs with {} sweeps".format(num_examples, nu))
        x = hm.setup.auto_setup.get_test_matrix(level.a, nu, num_examples=num_examples)
        _LOGGER.info("RER {:.3f}".format(norm(level.a.dot(x)) / norm(x)))

    # Create coarsening.
    coarsener, s = hm.repetitive.locality.create_coarsening(x, aggregate_size, num_components, normalize=False)
    r = coarsener.tile(level.a.shape[0] // aggregate_size)

    # Calculate local Mock cycle rates.
    level_subdomain = hm.setup.hierarchy.create_finest_level(a_subdomain)
    r_subdomain = coarsener.tile(m)
    mock_conv = [hm.setup.auto_setup.mock_cycle_conv_factor(level_subdomain, r_subdomain, nu) for nu in nu_values]

    # Interpolation by LS fitting for different calibers.
    calibers = (2, 3, 4)
    p = dict((caliber, hm.setup.auto_setup.create_interpolation(
        x, level.a, r, level.location, domain_size, interpolation_method, aggregate_size=aggregate_size,
        num_components=num_components,
        neighborhood=neighborhood, repetitive=repetitive, target_error=0.1,
        caliber=caliber, fit_scheme=fit_scheme, weighted=weighted)) for caliber in calibers)

    # Symmetrizing restriction for high-order P. 'calibers' must contain 4 for this to work.
    q = hm.repetitive.symmetry.symmetrize(r, level.a.dot(p[4]), aggregate_size, num_components)

    # Measure 2-level rates.
    l2c = []
    a_domain = level.a
    for _, caliber, restriction_type in coarsening_types:
        interpolation = p[caliber]
        if restriction_type == "pt":
            restriction = interpolation.T
        elif restriction_type == "r":
            restriction = r
        elif restriction_type == "q":
            restriction = q
        else:
            raise Exception("Unsupported restriction_type {}".format(restriction_type))
        ml = hm.repetitive.locality.create_two_level_hierarchy_from_matrix(
            a_domain, level.location, r, interpolation, restriction, aggregate_size, num_components, m=m)
        a = ml[0].a
        ac = ml[1].a
        fill_in_factor = (ac.nnz / a.nnz) * (a.shape[0] / ac.shape[0])
        symmetry_deviation = np.max(np.abs(ac - ac.transpose()))
        two_level_conv = [hm.repetitive.locality.two_level_conv_factor(
            ml, nu_pre, nu_post=0, nu_coarsest=nu_coarsest, print_frequency=print_frequency)[1] for nu_pre in nu_values]
        l2c.append([fill_in_factor, symmetry_deviation] + two_level_conv)

    data = np.array([[np.nan] * 2 + mock_conv] + l2c)
    all_conv = pd.DataFrame(np.array(data),
                            columns=("Fill-in", "Symmetry") + tuple(nu_values),
                            index=("Mock",) + tuple(item[0] for item in coarsening_types))
    return all_conv, r, p, q


def initial_tv(level, nu: int, ideal_tv: bool = False, num_examples: int = 5):
    if ideal_tv:
        _LOGGER.info("Generating {} ideal TVs".format(num_examples))
        x, lam = hm.analysis.ideal.ideal_tv(level.a, num_examples)
    else:
        _LOGGER.info("Generating {} TVs with {} sweeps".format(num_examples, nu))
        x = hm.setup.auto_setup.get_test_matrix(level.a, nu, num_examples=num_examples)
    return x


def improve_tv(x, multilevel, num_cycles: int = 1, print_frequency: int = None,
               nu_pre: int = 2, nu_post: int = 2, nu_coarsest: int = 4):
    # TODO(orenlivne): update parameters of relaxation cycle to reasonable values if needed.

    def relax_cycle(x):
        return hm.solve.relax_cycle.relax_cycle(multilevel, 1.0, nu_pre, nu_post, nu_coarsest).run(x)

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

    _LOGGER.info("Improving vectors by relaxation cycles ({}, {}; {})".format(nu_pre, nu_post, nu_coarsest))
    x, _ = hm.solve.run.run_iterative_method(level.operator, relax_cycle, x, num_cycles)
    return x


def build_coarse_level(level, x, domain_size: float,
                       aggregate_size: int, num_components: int,
                       interpolation_method: str = "ls",
                       fit_scheme: str = "ridge",
                       weighted: bool = False,
                       neighborhood: str = "extended",
                       repetitive: bool = False,
                       caliber: int = 2,
                       m: int = None):
    # Generate initial test vectors.
    if m is None:
        m = level.size // aggregate_size
    subdomain_size = m * aggregate_size
    a_subdomain = level.a[:subdomain_size, :subdomain_size]
    # if m is None:
    #     m = level.size // aggregate_size
    # _LOGGER.info("Domain size {}".format(m * aggregate_size))

    # Create coarsening.
    _LOGGER.info("Coarsening: aggregate_size {} num_components {}".format(aggregate_size, num_components))
    coarsener, s = hm.repetitive.locality.create_coarsening(x, aggregate_size, num_components, normalize=False)
    r = coarsener.tile(level.a.shape[0] // aggregate_size)

    # Calculate local Mock cycle rates. For now, this is performed over the entire domain.
    # TODO: make this local.
    nu_values = np.arange(1, 12)
    #level_subdomain = hm.setup.hierarchy.create_finest_level(a_subdomain)
    #r_subdomain = coarsener.tile(m)
    mock_conv = np.array([hm.setup.auto_setup.mock_cycle_conv_factor(level, r, nu) for nu in nu_values])
    print("mock cycle rate", mock_conv)
    print(x.shape, r.shape)

    # Interpolation by LS fitting.
    p = hm.setup.auto_setup.create_interpolation(
        x, level.a, r, level.location, domain_size, interpolation_method, aggregate_size=aggregate_size,
        num_components=num_components,
        neighborhood=neighborhood, repetitive=repetitive, target_error=0.1,
        caliber=caliber, fit_scheme=fit_scheme, weighted=weighted)

    q = p.T
    # Measure local 2-level rates. These are global rates.
    # TODO: make it local.
    ml = hm.repetitive.locality.create_two_level_hierarchy_from_matrix(
        level.a, level.location, r, p, q, aggregate_size, num_components, m=None)
    two_level_conv = np.array([hm.repetitive.locality.two_level_conv_factor(
        ml, nu, nu_coarsest=-1, print_frequency=None)[1] for nu in nu_values])
    print("2-level rate   ", two_level_conv)

    # Return the 2-level hierarchy, tiled over the entire domain.
    ml = hm.repetitive.locality.create_two_level_hierarchy_from_matrix(
        level.a, level.location, r, p, q, aggregate_size, num_components)
    return ml


def run_r_vs_q(level, z, r, p, q, aggregate_size, num_components, nu, nu_coarsest):
    titles = ("r", "q")
    restrictions = (r, q)
    fig, axs = plt.subplots(1, len(titles), figsize=(12, 4))

    for title, restriction, ax in zip(titles, restrictions, axs):
        print("Restriction", title)
        ml = hm.repetitive.locality.create_two_level_hierarchy_from_matrix(
            level.a, level.location, r, p, restriction, aggregate_size, num_components)
        y, _ = hm.repetitive.locality.two_level_conv_factor(
                    ml, nu_pre=0, nu_post=nu, nu_coarsest=nu_coarsest, print_frequency=1,
                    debug=False, z=z, seed=0, num_sweeps=15, num_levels=2)
        #y -= z.dot(z.T.dot(y[:, None])).flatten()

        # Asymptotic vector.
        ax.set_title("Slowest Vector in Two-level Cycle(0, {})".format(nu))
        e = ml[1].interpolate(r.dot(y))
        ax.plot(y, label="x");
        ax.plot(e, label="PRx");
        ax.grid(True)
        ax.legend()