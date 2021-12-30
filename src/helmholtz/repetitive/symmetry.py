import itertools
import helmholtz as hm
import helmholtz.repetitive.coarsening_repetitive as hrc
import numpy as np


def symmetrize(r, ap, aggregate_size, num_components):
    """
    Returns Q such that Q*A*P is symmetric and has the sparsity pattern of R*A*P in a repetitive framework.

    :param r: R.
    :param ap: A*P.
    :param aggregate_size: coarse aggregate size.
    :num_components: number of principal components = # coarse vars per aggregate.
    :return:
    """
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

    Q = hrc.Coarsener(q.reshape((num_components, aggregate_size))).tile(n // aggregate_size)
    return Q


# def update_interpolation(x, r, caliber, num_windows):
#     conv = [[np.nan] + [
#         hm.setup.auto_setup.mock_cycle_conv_factor(level, r, nu)
#         for nu in nu_values]]
#     p = hm.setup.auto_setup.create_interpolation(
#             x, level.a, r, level.location, domain_size, interpolation_method, aggregate_size=aggregate_size, num_components=num_components,
#             neighborhood=neighborhood, repetitive=repetitive, target_error=0.1,
#             caliber=caliber, fit_scheme=fit_scheme, weighted=weighted)
#     ml = hm.repetitive.locality.create_two_level_hierarchy_from_matrix(
#             level.a, level.location, r, p, aggregate_size, num_components, restriction=r)
#     ac = ml[1].a
#     fill_in_factor = (ac.nnz / ml[0].a.nnz) * (ml[0].a.shape[0] / ac.shape[0])
#     symmetry_deviation = np.max(np.abs(ac - ac.T))
#     two_level_conv = [
#         hm.repetitive.locality.two_level_conv_factor(ml, nu, print_frequency=None)[1]
#         for nu in nu_values]
#     conv.append([symmetry_deviation] + two_level_conv)
#     return ml, conv

# caliber = 4
# num_examples = 4
# num_windows = 32
# num_iterations = 10
# x = hm.setup.auto_setup.get_test_matrix(level.a, num_sweeps_on_vectors, num_examples=num_examples)

# all_conv = []
# r = coarse_level._r
# ml, conv = update_interpolation(x, r, caliber, num_windows)
# p = ml[1]._p
# all_conv += conv
# for i in range(num_iterations):
#     r = symmetrize(r, level.a.dot(p), num_components, aggregate_size)
#     ml, conv = update_interpolation(x, r, caliber, num_windows)
#     p = ml[1]._p
#     all_conv += conv

# all_conv = pd.DataFrame(all_conv,
#                     columns=("Symmetry",) + tuple(nu_values),
#                     index=sum((("{} Mock".format(i), "{} 2L".format(i))
#                         for i in range(num_iterations + 1)), ()))
# styler = all_conv.style.set_caption("Convergence Factors").format(precision=4)
# display_html(styler._repr_html_(), raw=True)