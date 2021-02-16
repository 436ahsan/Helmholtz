"""Linear algebra operations, sparse operator definitions."""
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.linalg import norm
from scipy.linalg import eig
from typing import Tuple


def normalize_signs(r, axis=0):
    """
    Multiplies r by a diagonal matrix with entries +1 or -1 so that the signs of the first row (or column)
    are all positive.

    Args:
        r:
        axis: axis to normalize signs along. If axis=0, normalizes rows to positive first element (i.e., the
        first column will be normalized to all positive numbers). If axis=1, normalizes columns instead.

    Returns: sign-normalized matrix.
    """
    if axis == 0:
        return r * np.sign(r[:, 0])[:, None]
    else:
        return r * np.sign(r[0])[None, :]


def get_window(x: np.ndarray, offset: int, aggregate_size: int) -> np.ndarray:
    """
    Returns a periodic window x[offset:offset+aggregate_size].

    Args:
        x: vector or matrix.
        offset: window start.
        aggregate_size: window size.

    Returns: x[offset:offset+aggregate_size]. Wraps around if out of bounds.
    """
    # Wrap around window since we're on a periodic domain.
    x_aggregate = x[offset:min(offset + aggregate_size, len(x))]
    if offset + aggregate_size > len(x):
        x_aggregate = np.concatenate((x_aggregate, x[:offset + aggregate_size - len(x)]), axis=0)
    return x_aggregate


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


def scaled_norm_of_matrix(e: np.ndarray) -> float:
    """
    Returns the scaled L2 norm of each test function e in a test matrix:

     [ sum(e[i1,...,id] ** 2 for all (i1,...,id)) / np.prod(e.shape) ] ** 0.5

    Args:
        e: test matrix, where e.shape[d] = #gridpoints in dimension d and e.shape[num_dims] = #test functions.

    Returns:
        The scaled L2 norm of e.
    """
    num_dims = e.ndim - 1
    return norm(e, axis=tuple(range(num_dims))) / np.prod(e.shape[:num_dims]) ** 0.5


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


def gram_schmidt(a: np.ndarray) -> np.ndarray:
    """
    Performs a Gram-Schmidt orthonormalization on matrix columns. Uses the QR factorization, which is more
    numerically stable than the explicit Gram-Schmidt algorithm.

    Args:
        a: original matrix.

    Returns: orthonormalized matrix a.
    """
    return scipy.linalg.qr(a, mode="economic")[0]


def ritz(x: np.ndarray, action) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs a Ritz projection on matrix columns.

    Args:
        x: original matrix (n x k).
        action: a functor of the action L(x) of the operator L (n x k matrix -> n x k matrix).

    Returns: tuple of (Ritz-projected matrix a, vector of corresponding eigenvalues).
    """
    # Make x an orthogonal projection onto the space X spanned by the original x columns.
    x = gram_schmidt(x)
    # Form the Gram matrix.
    g = x.transpose().dot(action(x))
    # Solve the eigenproblem g*z = lam*z for the coefficients z of the Ritz projection x*z.
    lam, z = eig(g)
    # Convert back to the original coordinates (x).
    lam = np.real(lam)
    ind = np.argsort(np.abs(lam))
    x_projected = x.dot(z)
    return x_projected[:, ind], lam[ind]
