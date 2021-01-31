"""Bootstrap AMG processes that generate test functions with low Helmholtz residuals on a periodic domain."""
import logging

import numpy as np
import scipy.sparse
from numpy.linalg import svd
from typing import Tuple

import helmholtz as hm

_LOGGER = logging.getLogger(__name__)


def generate_test_functions(a_coarsest: scipy.sparse.spmatrix, num_growth_steps: int,
                            growth_factor: int = 2,
                            num_bootstrap_steps: int = 1) -> Tuple[nd.ndarray, hm.multilevel.Multilevel]:
    """
    Creates low-residual test functions and multilevel hierarchy on a large domain from the operator on a small window
    (the coarsest domain). This is similar to a full multigrid algorithm.

    Args:
        a_coarsest: the operator on the coarsest domain.
        num_growth_steps: number of steps to increase the domain.
        growth_factor: by how much to increase the domain size at each growth step.
        num_bootstrap_steps: number of bootstrap steps to perform on each domain.

    Returns:
        x: test matrix on the largest (final) domain.
        multilevel: multilevel hierarchy on the largest (final) domain.
    """
    # Initialize test functions (to random) and hierarchy at coarsest level.
    multilevel = hm.multilevel.Multilevel()
    level = hm.multilevel.Level(a)
    multilevel.level.append(level)
    x = hm.multilevel.random_test_matrix(domain_shape, num_examples=num_examples)
    # Bootstrap at the current level.
    for i in range(num_bootstrap_steps):
        x, multilevel = bootstap(x, multilevel)

    for l in range(num_growth_steps):
        # Tile solution and hierarchy on the twice larger domain.
        x = hm.linalg.periodic_tile(x, growth_factor)
        multilevel = multilevel.periodic_tile(growth_factor)
        # Bootstrap at the current level.
        for i in range(num_bootstrap_steps):
            x, multilevel = bootstap(x, multilevel)

    return x, multilevel


class BootstrapMultilevelBuilder:
    """Builds a multilevel hierarchy on a fixed-size domain using bootstrapping."""

    def __init__(self, a: scipy.sparse.spmatrix):
        """
        Creates a bootstrap multilevel builder (setup) object.
        Args:
            a: finest-level operator.
        """
        self.multilevel = hm.multilevel.Multilevel()
        level = hm.multilevel.Level(a)
        self.multilevel.level.append(level)
        # Initialize test matrix to random.
        self.x = hm.multilevel.random_test_matrix(domain_shape, num_examples=num_examples)

    def relax_test_matrix(self, num_sweeps: int = 10000, print_frequency: int = None):
        b = np.zeros_like(x)
        self.x, conv_factor = hm.multilevel.relax_test_matrix(
            level.operator, lambda x: level.relax(x, b), self.x, num_sweeps=num_sweeps, print_frequency=print_frequency)
        return conv_factor


# In 1D, we know there are two principal components.
# TODO(oren): replace nc by a dynamic number in general based on # large singular values.
def create_coarse_level(level, x, aggregate_size: int = 4, num_examples: int = None, nc: int = 2):
    """
    Coarsens a level.
    Args:
        level:
        x:
        aggregate_size:
        num_examples:
        nc:

    Returns:

    """
    # TODO(oren): generalize the domain to the d-dimensional case. For now assuming 1D only.
    domain_size = a.shape[0]
    domain_shape = (domain_size, )
    aggregate_shape = (aggregate_size, )
    assert domain_size % aggregate_size == 0, \
        "Aggregate shape must divide the domain shape in every dimension"
    # assert all(ni % ai == 0 for ni, ai in zip(domain_shape, aggregate_shape)), \
    #     "Aggregate shape must divide the domain shape in every dimension"

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
    return coarse_level


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
