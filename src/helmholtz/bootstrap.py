"""Bootstrap AMG processes that generate test functions with low Helmholtz residuals on a periodic domain."""
import helmholtz as hm
import logging
import numpy as np
import scipy.sparse
from numpy.linalg import svd, norm
from helmholtz.linalg import scaled_norm

_LOGGER = logging.getLogger(__name__)


# In 1D, we know there are two principal components.
# TODO(oren): replace nc by a dynamic number in general based on # large singular values.
def generate_test_matrix(a: scipy.sparse.dia_matrix, aggregate_size: int = 4, num_examples: int = None,
                         nc: int = 2,
                         num_sweeps: int = 10000, print_frequency: int = None):
    # TODO(oren): generalize the domain to the d-dimensional case. For now assuming 1D only.
    domain_size = a.shape[0]
    domain_shape = (domain_size, )
    aggregate_shape = (aggregate_size, )
    assert domain_size % aggregate_size == 0, \
        "Aggregate shape must divide the domain shape in every dimension"
    # assert all(ni % ai == 0 for ni, ai in zip(domain_shape, aggregate_shape)), \
    #     "Aggregate shape must divide the domain shape in every dimension"

    # Generate initial test functions by 1-level relaxation, starting from random[-1, 1].
    if num_examples is None:
        # By default, use more test functions than gridpoints so we have a sufficiently large test function sample.
        num_examples = 3 * 4 * np.prod(aggregate_shape)

    multilevel = hm.multilevel.Multilevel()
    level = hm.multilevel.Level(a)
    multilevel.level.append(level)
    x = hm.multilevel.random_test_matrix(domain_shape, num_examples=num_examples)
    b = np.zeros_like(x)
    x = hm.multilevel.relax_test_matrix(level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps,
                                        print_frequency=print_frequency)
    x_aggregate_t = x[:aggregate_size].transpose()
    r, s = create_coarse_vars(x_aggregate_t, domain_size, nc)
    _LOGGER.debug("Singular values {}".format(s))
    xc = r.dot(x)
    xc_t = xc.transpose()
    caliber = 2
    p = create_interpolation(x_aggregate_t, xc_t, domain_size, nc, caliber)
    ac = (r.dot(a)).dot(p)
    coarse_level = hm.multilevel.Level(ac, r, p)
    multilevel.level.append(coarse_level)
    return x, multilevel


def create_coarse_vars(x_aggregate_t, domain_size: int, nc: int) -> scipy.sparse.csr_matrix:
    """Generates R (coarse variables). R of single aggregate = SVD principal components over an aggregate; global R =
    tiling of the aggregate R over the domain."""
    aggregate_size = x_aggregate_t.shape[1]
    _, s, vh = svd(x_aggregate_t)
    r = vh[:nc]
    # Tile R of a single aggregate over the entire domain.
    num_aggregates = domain_size // aggregate_size
    r = scipy.sparse.block_diag(tuple(r for _ in range(num_aggregates))).tocsr()
    return r, s


def create_interpolation(x_aggregate_t: np.ndarray, xc_t: np.ndarray, domain_size: int, nc: int, caliber: int) -> \
        scipy.sparse.csr_matrix:
    """Defines interpolation to an aggregate by LS fitting to coarse neighbors of each fine var. The global
    interpolation P is the tiling of the aggregate P over the domain."""

    # Define nearest coarse neighbors of each fine variable.
    aggregate_size = x_aggregate_t.shape[1]
    num_aggregates = domain_size // aggregate_size
    nbhr = hm.interpolation_nbhr.geometric_neighbors(domain_size, aggregate_size, nc)
    nbhr = hm.interpolation_nbhr.sort_neighbors_by_similarity(x_aggregate_t, xc_t, nbhr)

    # Fit interpolation over an aggregate.
    alpha = np.array([0, 0.001, 0.01, 0.1, 1.0])
    num_examples = x_aggregate_t.shape[0]
    fitter = hm.interpolation_fit.InterpolationFitter(
        x_aggregate_t, xc=xc_t, nbhr=nbhr,
        fit_samples=num_examples // 3, val_samples=num_examples // 3, test_samples=num_examples // 3)
    error, alpha_opt = fitter.optimized_relative_error(caliber, alpha, return_weights=True)
    p_aggregate = (np.tile(np.arange(aggregate_size)[:, None], nc).flatten(),
                   np.concatenate([nbhr_i for nbhr_i in nbhr]),
                   np.concatenate([pi for pi in error[:, 2:]]),)
    print("Interpolation error", error[:, 1])

    # Tile P of a single aggregate over the entire domain.
    row = np.concatenate([p_aggregate[0] + aggregate_size * ic for ic in range(num_aggregates)])
    col = np.concatenate([p_aggregate[1] + nc * ic for ic in range(num_aggregates)])
    data = np.tile(p_aggregate[2], num_aggregates)
    p = scipy.sparse.coo_matrix((data, (row, col)), shape=(domain_size, nc * num_aggregates)).tocsr()
    return p


def period_double(a):
    """Tiles the periodic B.C. operator on a twice-larger domain."""
    pass
