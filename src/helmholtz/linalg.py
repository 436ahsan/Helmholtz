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


def get_uniform_aggregate_starts(domain_size, aggregate_size):
    """
    Returns the list of aggregate starts, for a domain size and fixed aggregate size over the entire domain. The last
    two aggregates overlap if the domain size is not divisible by the aggregate size.
    Args:
        domain_size: domain size.
        aggregate_size: aggregate sixe.

    Returns: list of aggregate start indices.
    """
    return list(range(0, domain_size, aggregate_size)) if domain_size % aggregate_size == 0 else \
        list(range(0, domain_size - aggregate_size, aggregate_size)) + [domain_size - aggregate_size]


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
    Tiles a dense matrix (e.g., the coarsening R over an aggregate) over a domain of non-overlapping aggregates.
    Args:
        r: aggregate matrix.
        n: number of times to tile a.

    Returns: r, tiled n over n aggregates.
    """
    return scipy.sparse.block_diag(tuple(r for _ in range(n))).tocsr()


def helmholtz_1d_discrete_operator(kh: float, discretization: str, n: int) -> scipy.sparse.dia_matrix:
    """
    Returns the normalized FD-discretized 1D Helmholtz operator with periodic boundary conditions. The discretization
    stencil is [1, -2 + (kh)^2, -1].

    Args:
        kh: k*h, where k is a the wave number and h is the meshsize.
        discretization: "3-point"|"5-point", discretization scheme.
        n: size of grid.

    Returns:
        Helmholtz operator (as a sparse matrix).
    """
    if discretization == "3-point":
        return helmholtz_1d_operator(kh, n)
    elif discretization == "5-point":
        return helmholtz_1d_5_point_operator(kh, n)
    else:
        raise Exception("Unsupported discretization {}".format(discretization))


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


def helmholtz_1d_5_point_operator(kh: float, n: int) -> scipy.sparse.dia_matrix:
    """
    Returns the normalized FD-discretized 1D Helmholtz operator with periodic boundary conditions. This is an O(h^4)
    discretization with stencil is [1, -2 + (kh)^2, -1] / 12.

    Args:
        kh: k*h, where k is a the wave number and h is the meshsize.
        n: size of grid.

    Returns:
        Helmholtz operator (as a sparse matrix).
    """
    return sparse_circulant(np.array([-1, 16, -30 + 12 * kh ** 2, 16, -1]) / 12, np.array([-2, -1, 0, 1, 2]), n)


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


# TODO(orenlivne): replace by a sparse Cholesky A*A^T factorization, pass in A directly to it instead of forming A*A^T.
# explicitly and calculating its LU factorization. scipy doesn't provide an implementation; could not install
# scikit-sparse on the Mac (https://scikit-sparse.readthedocs.io/en/latest/cholmod.html).
class SparseLuSolver:
    """Performs a sparse LU factorization of a scipy matrix, converts the LU parts to sparse torch tensors so that
    they can be multiplied by."""

    def __init__(self, a):
        """Constructs a solver of A*x = b where A = sparse scipy matrix. Performs LU factorization."""
        n = a.shape[0]
        lu = scipy.sparse.linalg.splu(a.tocsc())
        self._l_inv = scipy.sparse.csc_matrix(
            scipy.sparse.linalg.spsolve_triangular(lu.L.tocsr(), np.identity(n)))
        self._u_inv = scipy.sparse.csc_matrix(
            scipy.sparse.linalg.spsolve_triangular(lu.U.transpose().tocsr(), np.identity(n))).transpose()
        self._pr = scipy.sparse.csc_matrix((np.ones(n), (lu.perm_r, np.arange(n))))
        self._pc = scipy.sparse.csc_matrix((np.ones(n), (np.arange(n), lu.perm_c)))

    def solve(self, b: np.ndarray):
        """Solves A*x = b. b is a tensor of shape a.shape[0] x k for some k."""
        # TODO(orenlivne): replace permutation matrix multiplications by a simple row/column indexing call.
        x1 = self._pr.dot(b)
        x2 = self._l_inv.dot(x1)
        x3 = self._u_inv.dot(x2)
        x = self._pc.dot(x3)
        return x


def pairwise_cos_similarity(x: np.ndarray, y: np.ndarray = None, squared: bool = False, center: bool = False):
    """
    Returns the pairwise cosine distance matrix between columns of two matrices.
    
    Args:
        x: x: n x p matrix.
        y: n x q matrix. Setting y= None is as the same as y = x.
        squared: if True, returns the cos distance squared.
        center: if True, removes the means from x and y, i.e., returns the correlation.

    Returns: matrix M of cosine distances of size p x q:

        M_{ij} = sum(x[:,i] * y[:,j]) / (sum(x[:,i] ** 2) * sum(y[:,j] ** 2) ** 0.5
    """
    if y is None:
        y = x
    if center:
        x -= np.mean(x, axis=0)[None, :]
        y -= np.mean(y, axis=0)[None, :]

    x2 = np.sum(x ** 2, axis=0)
    y2 = np.sum(y ** 2, axis=0)
    if squared:
        return (x.transpose().dot(y) ** 2) / np.clip((x2[:, None] * y2[None, :]), 1e-15, None)
    else:
        return x.transpose().dot(y) / np.clip((x2[:, None] * y2[None, :]) ** 0.5, 1e-15, None)


def csr_vappend(a: scipy.sparse.csr_matrix, b: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    """
    Takes in 2 csr_matrices and appends the second one to the bottom of the first one.
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied.

    Args:
        a: top matrix.
        b: bottom matrix.

    Returns: concatenated matrix.
    """
    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a
