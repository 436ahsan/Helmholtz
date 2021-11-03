"""Routines related to local coarsening derivation: local mock cycle rate, local 2-level cycle rate. Especially
useful for repetitive systems."""
import helmholtz as hm
import helmholtz.repetitive.coarsening_repetitive as hrc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NU_VALUES = np.arange(1, 7)
M_VALUES = np.arange(2, 11)


def mock_conv_factor(kh, n, aggregate_size, num_components, num_sweeps: int = 5, num_examples: int = 3,
                     nu_values: np.ndarray = NU_VALUES, m_values: np.ndarray = M_VALUES):
    # 'num_sweeps': number of relaxation sweeps to relax the TVs and to use in coarsening optimization
    # (determined by relaxation shrinkage).

    # Create fine-level matrix.
    a = hm.linalg.helmholtz_1d_operator(kh, n)
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


def mock_conv_factor_for_domain_size(kh, r, aggregate_size, m, nu_values):
    """Returns thre mock cycle conv factor for a domain of size m instead of n."""
    # Create fine-level matrix.
    a = hm.linalg.helmholtz_1d_operator(kh, m)
    # Use default Kacmzarz for kh != 0.
    level = hm.setup.hierarchy.create_finest_level(a, relaxer=hm.solve.relax.GsRelaxer(a) if kh == 0 else None)
    r_csr = r.tile(m // aggregate_size)
    return np.array([hm.setup.auto_setup.mock_cycle_conv_factor(level, r_csr, nu) for nu in nu_values])


def create_two_level_hierarchy(kh, m, r, p, aggregate_size):
    a = hm.linalg.helmholtz_1d_operator(kh, m)
    r_csr = hm.linalg.tile_array(r, m // aggregate_size)
    p_csr = hm.linalg.tile_array(p, m // aggregate_size)
    level0 = hm.setup.hierarchy.create_finest_level(a)
    # relaxer=hm.solve.relax.GsRelaxer(a) if kh == 0 else None)
    level1 = hm.setup.hierarchy.create_coarse_level(level0.a, level0.b, r_csr, p_csr)
    multilevel = hm.hierarchy.multilevel.Multilevel.create(level0)
    multilevel.add(level1)
    return multilevel


def two_level_conv_factor(multilevel, nu, print_frequency: int = None):
    level = multilevel.finest_level
    n = level.size
    # Test two-level cycle convergence for A*x=b with random b.
    b = np.random.random((n, ))

    def two_level_cycle(y):
        return hm.solve.solve_cycle.solve_cycle(multilevel, 1.0, nu, 0, nu_coarsest=-1, debug=False, rhs=b).run(y)

    def residual(x):
        return b - multilevel[0].operator(x)

    return hm.solve.run.run_iterative_method(
        residual, two_level_cycle, np.random.random((n, )), 15, print_frequency=print_frequency)


def two_level_conv_data_frame(kh, r, p, aggregate_size, m_values, nu_values):
    return pd.DataFrame(np.array([
        [two_level_conv_factor(create_two_level_hierarchy(kh, m * aggregate_size, r, p, aggregate_size), nu)[1]
         for nu in nu_values]
        for m in m_values]),
            index=m_values, columns=nu_values)


def create_coarsening(x, aggregate_size, num_components):
    # Construct coarsening on an aggregate.
    x_aggregate_t = np.concatenate(
        tuple(hm.linalg.get_window(x, offset, aggregate_size)
              for offset in range(max((4 * aggregate_size) // x.shape[1], 1))), axis=1).transpose()
    # Tile the same coarsening over all aggregates.
    r, s = hm.setup.coarsening_uniform.create_coarsening(x_aggregate_t, num_components)
    print("Coarsening:", "a", aggregate_size, "nc", num_components, "#windows", x_aggregate_t.shape[0], "r", r, "s", s)
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
