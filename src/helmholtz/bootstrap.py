"""Bootstrap AMG processes that generate test functions with low Helmholtz residuals on a periodic domain."""
import logging

import numpy as np
import scipy.sparse
from numpy.linalg import svd
from typing import Tuple

import helmholtz as hm

_LOGGER = logging.getLogger(__name__)


def generate_test_functions(a: scipy.sparse.spmatrix, num_growth_steps: int,
                            growth_factor: int = 2,
                            num_bootstrap_steps: int = 1,
                            aggregate_size: int = 4) -> Tuple[np.ndarray, hm.multilevel.Multilevel]:
    """
    Creates low-residual test functions and multilevel hierarchy on a large domain from the operator on a small window
    (the coarsest domain). This is similar to a full multigrid algorithm.

    Args:
        a: the operator on the coarsest domain.
        num_growth_steps: number of steps to increase the domain.
        growth_factor: by how much to increase the domain size at each growth step.
        num_bootstrap_steps: number of bootstrap steps to perform on each domain.
        aggregate_size: aggregate size = #fine vars per aggregate

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
        _LOGGER.info("Growing domain to level {}, size {}".format(l, growth_factor * x.shape[0]))
        # Tile solution and hierarchy on the twice larger domain.
        x = hm.linalg.periodic_tile(x, growth_factor)
        multilevel = multilevel.periodic_tile(growth_factor)
        # Bootstrap at the current level.
        for i in range(num_bootstrap_steps):
            x, multilevel = bootstap(x, multilevel, aggregate_size=aggregate_size)

    return x, multilevel


def bootstap(x, multilevel, aggregate_size: int = 4, nc: int = 2):
    """
    Improves test functions and a multilevel hierarchy on a fixed-size domain by bootstrapping.
    Args:
        x: test matrix.
        multilevel: multilevel hierarchy.
        aggregate_size: aggregate size = #fine vars per aggregate.
        nc: number of coarse variables per aggregate. In 1D at the finest level, we know there are two principal
        components, so nc=2 makes sense there.

    Returns:
        improved x, multilevel hierarchy with the same number of levels.
    """
    # At the finest level, use a multilevel cycle to improve x.
    level = multilevel.level[0]
    num_levels = len(multilevel)
    b = np.zeros_like(x)
    # TODO(orenlivne): update parameters of relaxation cycle to reasonable values if needed.
    relax_cycle = lambda x: multilevel.relax(x, 2, 2, 4) if num_levels > 1 else level.relax
    x, _ = hm.multilevel.relax_test_matrix(level.operator, relax_cycle, x, 100)

    # Recreate all coarse levels. One down-pass, relaxing at each level, hopefully starting from improved x so the
    # process improves all levels.
    # TODO(orenlivne): add nested bootstrap cycles if needed.
    new_multilevel = hm.multilevel.Multilevel()
    new_multilevel.append(level)
    for l in range(num_levels):
        _LOGGER.info("Coarsening level {}->{}".format(l - 1, l))
        level = create_coarse_level(level, x, aggregate_size=aggregate_size, num_examples=num_examples, nc=nc)
        new_multilevel.append(level)
        x = level.r.dot(x)
        x = hm.multilevel.relax_test_matrix(
            level.operator, lambda x: level.relax(x, b), x, num_sweeps=num_sweeps, print_frequency=print_frequency)
    return x, new_multilevel


# TODO(oren): replace nc by a dynamic number in general based on # large singular values.
def create_coarse_level(level, x, aggregate_size: int = 4, nc: int = 2, caliber: int = 2):
    """
    Coarsens a level.
    Args:
        level: fine level to be coarsened.
        x: fine-level test matrix.
        aggregate_size: aggregate size = #fine vars per aggregate
        nc: number of coarse variables per aggregate. In 1D at the finest level, we know there are two principal
        components, so nc=2 makes sense there.
        caliber: interpolation caliber.

    Returns: coarse level obtained by fitting P, R to x.
    """
    # TODO(oren): generalize the domain to the d-dimensional case. For now assuming 1D only.
    domain_size = a.shape[0]
    assert domain_size % aggregate_size == 0, \
        "Aggregate shape must divide the domain shape in every dimension"
    # assert all(ni % ai == 0 for ni, ai in zip(domain_shape, aggregate_shape)), \
    #     "Aggregate shape must divide the domain shape in every dimension"

    x_aggregate_t = x[:aggregate_size].transpose()
    r, s = create_coarse_vars(x_aggregate_t, domain_size, nc)
    _LOGGER.debug("Singular values {}".format(s))
    xc = r.dot(x)
    xc_t = xc.transpose()
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
    r = hm.linalg.tile_dense(r, domain_size // aggregate_size)
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
