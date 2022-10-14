"""Routines related to local coarsening derivation: local mock cycle rate, local 2-level cycle rate. Especially
useful for repetitive systems."""
import helmholtz as hm
import helmholtz.repetitive.coarsening_repetitive as hrc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse

NU_VALUES = np.arange(1, 7)
M_VALUES = np.arange(2, 11)


def mock_conv_factor(kh, discretization, n, aggregate_size, num_components, num_sweeps: int = 5, num_examples: int = 3,
                     nu_values: np.ndarray = NU_VALUES, m_values: np.ndarray = M_VALUES):
    # 'num_sweeps': number of relaxation sweeps to relax the TVs and to use in coarsening optimization
    # (determined by relaxation shrinkage).

    # Create fine-level matrix.
    a = hm.linalg.helmholtz_1d_discrete_operator(kh, discretization, n)
    # Use default Kacmzarz for kh != 0.
    level = hm.setup.hierarchy.create_finest_level(a, relaxer=hm.solve.relax.GsRelaxer(a) if kh == 0 else None)
    # For A*x=b cycle tests.
    b = np.random.random((a.shape[0], ))

    # Create relaxed vectors.
    x = hm.solve.run.random_test_matrix((a.shape[0],), num_examples=num_examples)
    b = np.zeros_like(x)
    x, _ = hm.solve.run.run_iterative_method(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps)
    # import helmholtz.analysis.ideal
    # x, _ = helmholtz.analysis.ideal.ideal_tv(a, 2)
    return mock_conv_factor_for_test_vectors(kh, x, aggregate_size, num_components,
                                             nu_values=nu_values, m_values=m_values)


def mock_conv_factor_for_test_vectors(kh, x, aggregate_size, num_components,
                                      nu_values: np.ndarray = NU_VALUES, m_values: np.ndarray = M_VALUES):
    # Construct coarsening.
    r, s = create_coarsening(x, aggregate_size, num_components)

    # Calculate mock cycle convergence rate.
    mock_conv = pd.DataFrame(np.array([
        mock_conv_factor_for_domain_size(kh, r, aggregate_size, m * aggregate_size, nu_values)
        for m in m_values]),
            index=m_values, columns=nu_values)

    return r, mock_conv


def mock_conv_factor_for_domain_size(kh, discretization, r, aggregate_size, m, nu_values):
    """Returns thre mock cycle conv factor for a domain of size m instead of n."""
    # Create fine-level matrix.
    a = hm.linalg.helmholtz_1d_discrete_operator(kh, discretization, m)
    # Use default Kacmzarz for kh != 0.
    level = hm.setup.hierarchy.create_finest_level(a, relaxer=hm.solve.relax.GsRelaxer(a) if kh == 0 else None)
    r_csr = r.tile(m // aggregate_size)
    return np.array([hm.setup.auto_setup.mock_cycle_conv_factor(level, r_csr, nu) for nu in nu_values])


def create_two_level_hierarchy(kh, discretization, m, r, p, q, aggregate_size, nc):
    a = hm.linalg.helmholtz_1d_discrete_operator(kh, discretization, m)
    if isinstance(r, scipy.sparse.csr_matrix):
        r_csr = r
    else:
        r_csr = hm.linalg.tile_array(r, m // aggregate_size)
    if isinstance(p, scipy.sparse.csr_matrix):
        p_csr = p
    else:
        p_csr = hm.linalg.tile_array(p, m // aggregate_size)
    level0 = hm.setup.hierarchy.create_finest_level(a)
    level0.location = np.arange(a.shape[0])
    # relaxer=hm.solve.relax.GsRelaxer(a) if kh == 0 else None)
    level1 = hm.setup.hierarchy.create_coarse_level(level0.a, level0.b, r_csr, p_csr, q)
    # Calculate coarse-level variable locations. At each aggregate center we have 'num_components' coarse variables.
    level1.location = hm.setup.geometry.coarse_locations(level0.location, aggregate_size, nc)

    multilevel = hm.hierarchy.multilevel.Multilevel.create(level0)
    multilevel.add(level1)
    return multilevel


# TODO(orenlivne): clean this so that the callers pass in all objects for a problem on a fixed domain of size
# m * aggregate_size. This may be a sub-domain for locality in a repetitive framework.
def create_two_level_hierarchy_from_matrix(a, location, r, p, q, aggregate_size, num_components, symmetrize: bool = False,
                                           m: int = None):
    """

    :param a:
    :param location:
    :param r:
    :param p:
    :param q:
    :param aggregate_size:
    :param num_components:
    :param symmetrize:
    :param m: size of domain in aggregates. If None, m = a.shape[0] // aggregate_size.
    :return: two-level hierarchy.
    """
    if m is None:
        m = a.shape[0] // aggregate_size
    # Use the first m fine-level points as the sub-domain.

    r_csr = r if isinstance(r, scipy.sparse.csr_matrix) else hm.linalg.tile_array(r, m)
    p_csr = p if isinstance(p, scipy.sparse.csr_matrix) else hm.linalg.tile_array(p, m)
    q_csr = q if isinstance(q, scipy.sparse.csc_matrix) else hm.linalg.tile_array(q, m)

    n = m * aggregate_size
    nc = (n * num_components) // aggregate_size
    a = a[:n, :n]
    location = location[:n]
    r_csr = r_csr[:nc, :n]
    p_csr = p_csr[:n, :nc]
    q_csr = q_csr[:nc, :n]
    level0 = hm.setup.hierarchy.create_finest_level(a)
    level0.location = location
    # relaxer=hm.solve.relax.GsRelaxer(a) if kh == 0 else None)
    level1 = hm.setup.hierarchy.create_coarse_level(level0.a, level0.b, r_csr, p_csr, q_csr, symmetrize=symmetrize)
    # Calculate coarse-level variable locations. At each aggregate center we have 'num_components' coarse variables.
    level1.location = hm.setup.geometry.coarse_locations(level0.location, aggregate_size, num_components)

    multilevel = hm.hierarchy.multilevel.Multilevel.create(level0)
    multilevel.add(level1)
    return multilevel


def two_level_conv_factor(multilevel, nu_pre, nu_post: int = 0, nu_coarsest: int = -1,
                          print_frequency: int = None, debug: bool = False,
                          residual_stop_value: float = 1e-10, z: np.ndarray = None,
                          num_sweeps: int = 20, seed: int = None, num_levels: int = None):
    if seed is not None:
        np.random.seed(seed)
    level = multilevel.finest_level
    n = level.size
    # Test two-level cycle convergence for A*x=b with b=A*x0, x0=random[-1, 1].
    x_exact = 2 * np.random.random((n, )) - 1
    x_exact = np.zeros_like(x_exact)
    if z is not None:
        x_exact -= z.dot(z.T.dot(x_exact[:, None])).flatten()
    b = multilevel[0].operator(x_exact)
    #b = np.random.random((n, ))
    #b = np.ones((n, ))

    def two_level_cycle(y):
        return hm.solve.solve_cycle.solve_cycle(
            multilevel, 1.0, nu_pre, nu_post, nu_coarsest=nu_coarsest, debug=debug, rhs=b, num_levels=num_levels).run(y)

    def residual(x):
        return b - multilevel[0].operator(x)

    x_init = np.random.random((n, ))
    if z is not None:
        x_init -= z.dot(z.T.dot(x_init[:, None])).flatten()
    x, conv = hm.solve.run.run_iterative_method(
        residual, two_level_cycle, x_init, num_sweeps, print_frequency=print_frequency,
        residual_stop_value=residual_stop_value, z=z, x_exact=x_exact)
    return x_exact - x, conv


def two_level_conv_data_frame(kh, discretization, r, p, aggregate_size, m_values, nu_values):
    return pd.DataFrame(np.array([
        [two_level_conv_factor(create_two_level_hierarchy(kh, discretization, m * aggregate_size, r, p, aggregate_size), nu)[1]
         for nu in nu_values]
        for m in m_values]),
            index=m_values, columns=nu_values)


def create_coarsening(x, aggregate_size, num_components, normalize: bool = False, num_windows: int = None):
    # Construct coarsening on an aggregate.
    if num_windows is None:
        num_windows = 12 * aggregate_size
    x_aggregate_t = hm.linalg.get_windows_by_index(
        x, np.arange(aggregate_size), aggregate_size, num_windows)
    # Tile the same coarsening over all aggregates.
    r, s = hm.setup.coarsening_uniform.create_coarsening(x_aggregate_t, num_components, normalize=normalize)
    return hrc.Coarsener(r), s


def plot_coarsening(r, x):
    xc = r.dot(x)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    ax = axs[0]
    for i in range(2):
        ax.plot(x[:, i])
    ax.set_title("$x$")
    ax.grid(True)

    ax = axs[1]
    for i in range(2):
        ax.plot(xc[::2, i])
    ax.set_title("$x^c$ Species 0")
    ax.grid(True)

    ax = axs[2]
    for i in range(2):
        ax.plot(xc[1::2, i])
    ax.set_title("$x^c$ Species 1")
    ax.grid(True)
