import helmholtz as hm
import helmholtz.repetitive.coarsening_repetitive as hrc
import logging
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

_LOGGER = logging.getLogger(__name__)


def plot_svd_coarsening_accuracy(level, num_sweeps: int, aggregate_size: int, num_components):
    """Checks coarsening based on relaxed TVs, for different number of relaxation sweeps.
    If aggregate_size is not None, forces that aggregate size for the entire domain."""
    num_sweeps_values = 5 * 2 ** np.arange(8)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))

    # Create relaxed TVs.
    x_random = hm.solve.run.random_test_matrix((level.a.shape[0],), num_examples=4)
    b = np.zeros_like(x_random)
    x = hm.solve.run.run_iterative_method(
        level.operator, lambda x: level.relax(x, b), x_random, num_sweeps=num_sweeps)[0]
    r, s = hm.repetitive.locality.create_coarsening(x, aggregate_size, num_components, normalize=False)

    # x_random = hm.solve.run.random_test_matrix((a.shape[0],), num_examples=4 * aggregate_size)
    # b = np.zeros_like(x_random)
    # x = hm.solve.run.run_iterative_method(
    #    level.operator, lambda x: level.relax(x, b), x_random, num_sweeps=num_sweeps)[0]
    # start, end = 0, aggregate_size
    # x_aggregate_t = x[start:end].T
    # print(x_aggregate_t.shape)
    # r, s = hm.repetitive.coarsening_repetitive.create_coarsening(x_aggregate_t, threshold)
    r = r.asarray()

    # Relaxed vectors.
    ax = axs[0]
    for i in range(min(3, x.shape[1])):
        ax.plot(x[:, i]);
    ax.grid(True)
    ax.set_title(r"Test Vectors, $\nu={}$ sweeps".format(num_sweeps))

    ax = axs[1]
    # R should be real-valued, but cast just in case.
    for i, ri in enumerate(np.real(r)):
        ax.plot(ri)
    ax.set_title(r"Agg Size {} $n_c$ {}".format(r.shape[1], r.shape[0]))
    ax.set_ylabel(r"$R$ rows")
    ax.grid(True);

    # Singular values, normalized to sigma_max = 1.
    ax = axs[2]
    ax.plot(s / s[0], "rx")
    ax.set_title("Singular Values")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\sigma_k$")
    ax.grid(True);

    # TODO: replace by local mock cycle rate.
    nu_values = np.arange(1, 12)
    R = hrc.Coarsener(r).tile(level.a.shape[0] // aggregate_size)
    print("nu", "{:3d}".format(num_sweeps), "s", s / s[0], "Energy error", (1 - np.cumsum(s ** 2) / sum(s ** 2)) ** 0.5)
    mock_conv = np.array([hm.setup.auto_setup.mock_cycle_conv_factor(level, R, nu) for nu in nu_values])
    # hm.repetitive.locality.mock_conv_factor_for_domain_size(kh, discretization, r, aggregate_size, m * aggregate_size, nu_values)
    _LOGGER.info("Mock cycle conv {}".format(np.array2string(mock_conv, precision=3)))

#         # Generate coarse variables (R) on the non-repetitive domain.
#         r, aggregates, nc, energy_error = hm.repetitive.coarsening_repetitive.create_coarsening_domain(
#             x, threshold=threshold, fixed_aggregate_size=aggregate_size)
#         _LOGGER.info("Agg {}".format(np.array([len(aggregate) for aggregate in aggregates])))
#         _LOGGER.info("nc  {}".format(nc))
#         _LOGGER.info("Energy error mean {:.4f} max {:.4f}".format(np.mean(energy_error), np.max(energy_error)))
#         mock_conv_factor = np.array(
#             [hm.setup.auto_setup.mock_cycle_conv_factor(level, r, nu) for nu in np.arange(1, 16, dtype=int)])
#         _LOGGER.info("Mock cycle conv factor {}".format(np.array2string(mock_conv_factor, precision=3)))
    return r

def animate_shrinkage(factor, num_sweeps, residual, conv, rer, relax_conv_factor, x_history, r_history ):
    """Generatesanimation video of Kaczmarz shrinkage."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    duration = 3
    num_frames = len(r_history)
    meshsize = duration / num_frames

    # print(duration, num_frames, meshsize)

    def make_frame(t):
        i = min(int(np.round(t / meshsize)), len(r_history) - 1)
        ax = axs[0]
        ax.clear()
        color = "blue"
        x_ticks = np.arange(1, len(conv) + 1)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, len(conv)])
        ax.plot(x_ticks[:(i + 1)], conv[:(i + 1)], "o",
                color=color)  # , label=r"{} $\mu = {:.2f}, i = {}$".format(title, factor, num_sweeps))
        if i >= num_sweeps:
            ax.scatter([num_sweeps], [conv[num_sweeps - 1]], 120, facecolors='none', edgecolors=color)
        ax.set_ylabel("Residual Reduction Factor")
        ax.set_xlabel("Sweep #")
        ax.grid(True)

        ax = axs[1]
        ax.clear()
        ax.plot(x_history[i][:, 0])
        x_init = x_history[0][:, 0]
        ax.set_ylim([min(x_init) - 0.01, max(x_init) + 0.01])
        ax.set_title("Error after {} sweeps".format(i))

        ax = axs[2]
        ax.clear()
        ax.plot(r_history[i][:, 0])
        r_init = r_history[0][:, 0]
        ax.set_ylim([min(r_init) - 0.01, max(r_init) + 0.01])
        ax.set_title("Residual after {} sweeps".format(i))

        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=duration)
    return animation