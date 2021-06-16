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
    # Form Galerkin coarse-level operator.
    ac = (r.dot(a)).dot(p)
    bc = (r.dot(b)).dot(p)
    relaxer = hm.solve.relax.KaczmarzRelaxer(ac, bc)
    return hm.hierarchy.multilevel.Level(ac, bc, relaxer, r, p, r, p)


def create_finest_level(a: scipy.sparse.spmatrix) -> multilevel.Level:
    """
    Creates a repetitive domain finest level.
    Args:
        a: fine-level operator (stiffness matrix).

    Returns: finest level object.
    """
    b = scipy.sparse.eye(a.shape[0])
    relaxer = hm.solve.relax.KaczmarzRelaxer(a, b)
    return hm.hierarchy.multilevel.Level.create_finest_level(a, relaxer)
