import itertools
import helmholtz as hm
import helmholtz.repetitive.coarsening_repetitive as hrc
import numpy as np


def symmetrize(r, ap, num_components, aggregate_size):
    n, nc = ap.shape
    Q = r[:num_components]
    ap_cols = {}
    lhs, rhs = [], []
    for I in range(num_components):
        i = r[I].nonzero()[1]
        J = np.unique(ap[i].nonzero()[1])
        J_wrapped = hm.linalg.wrap_index_to_low_value(J, nc).astype(int)
        ap_J = ap[i][:, J]
        d = dict(((I, JJ), np.array(ap_J[:, col].todense()).flatten()) for col, JJ in enumerate(J_wrapped))
        ap_cols = dict(itertools.chain(ap_cols.items(), d.items()))
        lower = J_wrapped < I
        J_normalized = J_wrapped[lower] % num_components
        lhs += [(I, JJ) for JJ in J_wrapped[lower]]
        rhs += list(zip(J_normalized, I + J_normalized - J_wrapped[lower]))

    # Form symmetry equations C*q = (A - B)*q = 0.
    # Relying on the fact that q is stored as a compressed-row matrix (row after row in q.data).
    A = np.zeros((len(lhs), Q.nnz))
    for row, key in enumerate(lhs):
        A[row, Q.indptr[key[0]]:Q.indptr[key[0] + 1]] = ap_cols[key]
    B = np.zeros((len(rhs), Q.nnz))
    for row, key in enumerate(rhs):
        B[row, Q.indptr[key[0]]:Q.indptr[key[0] + 1]] = ap_cols[key]
    C = A - B

    # Kaczmarz (exact solver) on C*q = 0.
    # q <- q - C^T*M^{-1}*C*q where M = tril(C*C^T) is lex order Kaczmarz.
    q = Q.data.copy()
    q -= C.T.dot(np.linalg.solve(C.dot(C.T), C.dot(q)))

    R = hrc.Coarsener(q.reshape((num_components, aggregate_size))).tile(n // aggregate_size)
    return R