"""Utilities: multilevel hierarchy for non-repetitive problems."""
import scipy.sparse
import helmholtz as hm
import helmholtz.hierarchy.multilevel as multilevel


def create_coarse_level(a: scipy.sparse.csr_matrix,
                        b: scipy.sparse.csr_matrix,
                        r: scipy.sparse.csr_matrix,
                        p: scipy.sparse.csr_matrix,
                        q: scipy.sparse.csr_matrix,
                        symmetrize: bool = False) -> multilevel.Level:
    """
    Creates a tiled coarse level.
    Args:
        a: fine-level operator (stiffness matrix).
        b: fine-level mass matrix.
        r: coarsening operator.
        p: interpolation operator.
        q: residual restriction operator. If None, uses Q=P^T.
        symmetrize: iff True, use the symmetric part of the coarse-level operator.

    Returns: coarse level object.
    """
    # Form the SYMMETRIC Galerkin coarse-level operator.
    ac = (q.dot(a)).dot(p)
    bc = (q.dot(b)).dot(p)
    if symmetrize:
        ac = 0.5 * (ac + ac.T)
        bc = 0.5 * (bc + bc.T)
    relaxer = hm.solve.relax.KaczmarzRelaxer(ac, bc)
    return hm.hierarchy.multilevel.Level(ac, bc, relaxer, r, p, q)


def create_finest_level(a: scipy.sparse.spmatrix, relaxer=None) -> multilevel.Level:
    """
    Creates a repetitive domain finest level.
    Args:
        a: fine-level operator (stiffness matrix).
        relaxer: optional relaxation scheme. Defaults to Kaczmarz.

    Returns: finest level object.
    """
    b = scipy.sparse.eye(a.shape[0])
    if relaxer is None:
        relaxer = hm.solve.relax.KaczmarzRelaxer(a, b)
    return multilevel.Level.create_finest_level(a, relaxer)
