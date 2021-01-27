"""Bootstrap AMG processes that generate test functions with low Helmholtz residuals on a periodic domain."""
import helmholtz as hm
import logging
import numpy as np
import scipy.sparse
from numpy.linalg import svd, norm
from helmholtz.linalg import scaled_norm

logger = logging.getLogger(__name__)


def generate_test_functions(a: scipy.sparse.dia_matrix, aggregate_shape: int = 4, num_examples: int = None):
    relaxer = hm.kaczmarz.KaczmarzRelaxer(a)

    # In 1D, we know there are two principal components.
    # TODO(oren): replace this by a dynamic number in general based on # large singular values.
    num_aggregate_coarse_vars = 2

    # Generate initial test functions by 1-level relaxation, starting from random[-1, 1].
    num_sweeps = 10
    if num_examples is None:
        # By default, use more test functions than gridpoints so we have a sufficiently large test function sample.
        num_examples = 4 * np.prod(aggregate_shape)
    e = 2 * np.random.random((num_examples,) + aggregate_shape) - 1
    logger.debug("{:5d} |e| {:.8e} |r| {:.8e}".format(0, scaled_norm(e[:, 0]), scaled_norm(a.dot(e[:, 0]))))
    for i in range(1, num_sweeps + 1):
        e = relaxer.step(e)
        # A poor way of getting the last "column" of the tensor e.
        e0 = e.reshape(-1, e.shape[-1])[:, 0].reshape(e.shape[:-1])
        logger.debug("{:5d} |e| {:.8e} |r| {:.8e}".format(0, scaled_norm(e[:, 0]), scaled_norm(a.dot(e[:, 0]))))
    # Scale e to unit norm to avoid underflow, as we are calculating approximate eigenvectors of A.
    e /= norm(e)

    # Generate R (coarse variables). Columns = SVD principcal components over the aggregate.
    e_matrix = np.reshape(e, [np.prod(aggregate_shape), num_examples])
    _, s, vh = svd(e_matrix.transpose())
    print(s)
    r = vh[:num_aggregate_coarse_vars].transpose()
    print(r.shape)

    # Define interpolation by LS fitting.

    return e
