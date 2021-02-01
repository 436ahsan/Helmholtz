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


def tile_csr_matrix(a: scipy.sparse.csr_matrix, n: int) -> scipy.sparse.csr_matrix:
    """
    Tiles the periodic B.C. operator on a n-times larger domain.

    Args:
        a: sparse matrix on window.
        n: number of times to tile a.

    Returns:
        a on an n-times larger periodic domain.
    """
    n_row, n_col = a.shape
    row, col = a.nonzero()
    data = a.data
    # Calculate the positions of stencil neighbors relative to the stencil center.
    relative_col = col - row
    relative_col[relative_col >= n_col // 2] -= n_col
    relative_col[relative_col < -(n_col // 2)] += n_col

    # Tile the data into the ranges [0..n_col-1],[n_col,...,2*n_col-1],...[(n-1)*n_col,...,n*n_col-1].
    tiled_data = np.tile(data, n)
    tiled_row = np.concatenate([row + i * n_col for i in range(n)])
    tiled_col = np.concatenate([(row + relative_col + i * n_col) % (n * n_col) for i in range(n)])
    return scipy.sparse.coo_matrix((tiled_data, (tiled_row, tiled_col)), shape=(n * n_row, n * n_col)).tocsr()


def tile_array(r: np.ndarray, n: int) -> scipy.sparse.csr_matrix:
    """
    Tiles a dense matrix (e.g., the restriction R over an aggregate) over a domain of non-overlapping aggregates.
    Args:
        r: aggregate matrix.
        n: number of times to tile a.

    Returns: r, tiled n over n aggregates.
    """
    return scipy.sparse.block_diag(tuple(r for _ in range(n))).tocsr()


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
