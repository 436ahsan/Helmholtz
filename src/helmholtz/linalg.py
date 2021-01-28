"""Linear algebra operations, sparse operator definitions."""
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.linalg import norm


def scaled_norm(e: np.ndarray) -> float:
    """
    Returns the scaled L2 norm of a test function e:

     [ sum(e[i1,...,id] ** 2 for all (i1,...,id)) / np.prod(e.shape) ] ** 0.5

    Args:
        e: test function, where e.shape[d] = #gridpoints in dimension d.

    Returns:
        The scaled L2 norm of e.
    """
    return norm(e) / np.prod(e.shape) ** 0.5


def sparse_circulant(vals: np.array, offsets: np.array, n: int) -> scipy.sparse.dia_matrix:
    """
    Creates a sparse square circulant matrix from a stencil.
    Args:
        vals: stencil values.
        offsets: corresponding diagonal offsets. 0 corresponds to the middle of the stencil.
        n: matrix dimension.

    Returns:
        n x n sparse matrix with vals[i] on diagonal offsets[i] (wrapping around other diagonals -- n + offsets[i] or
        to -n + offsets[i] -- for periodic boundary conditions/circularity).
    """
    o = offsets[offsets != 0]
    v = vals[offsets != 0]
    dupoffsets = np.concatenate((offsets, n + o))
    dupvals = np.concatenate((vals, v))
    dupoffsets[dupoffsets > n] -= 2 * n

    return scipy.sparse.diags(dupvals, dupoffsets, shape=(n, n))


def helmholtz_1d_operator(kh: float, n: int) -> scipy.sparse.dia_matrix:
    """
    Returns the normalized FD-discretized 1D Helmholtz operator with periodic boundary conditions. The discretization
    stencil is [1, -2 + (kh)^2, -1].

    Args:
        kh: k*h, where k is a the wave number and h is the meshsize.
        n: size of grid.

    Returns:
        Helmholtz operator (as a sparse matrix).
    """
    return sparse_circulant(np.array([1, -2 + kh ** 2, 1]), np.array([-1, 0, 1]), n)
