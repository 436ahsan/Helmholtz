"""Utilities: multilevel hierarchy for non-repetitive problems."""
import scipy.sparse
import helmholtz as hm
import helmholtz.hierarchy.multilevel as multilevel


def create_coarse_level(a: scipy.sparse.csr_matrix, b: scipy.sparse.csr_matrix,
                        r: scipy.sparse.csr_matrix, p: scipy.sparse.csr_matrix) -> multilevel.Level:
    """
    Creates a tiled coarse level.
    Args:
        a: fine-level operator (stiffness matrix).
        b: fine-level mass matrix.
        r: aggregate coarsening.
        p: aggregate interpolation.

    Returns: coarse level object.
    """
    # Form the SYMMETRIC Galerkin coarse-level operator.
    pt = p.transpose()
    ac = (pt.dot(a)).dot(p)
    bc = (pt.dot(b)).dot(p)
    relaxer = hm.solve.relax.KaczmarzRelaxer(ac, bc)
    return hm.hierarchy.multilevel.Level(ac, bc, relaxer, r, p, r, p)


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
