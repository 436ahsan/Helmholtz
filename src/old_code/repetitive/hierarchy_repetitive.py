"""Utilities: multilevel hierarchy for repetitive problems."""
import scipy.sparse
import helmholtz as hm
import helmholtz.hierarchy.multilevel as multilevel


def create_tiled_coarse_level(a: scipy.sparse.spmatrix, b: scipy.sparse.spmatrix, r, p,
                              use_r_as_restriction: bool = False) -> multilevel.Level:
    # r: hm.repetitive.coarsening_repetitive.Coarsener,
    # p: hm.repetitive.interpolation_repetitive.Interpolator) -> \
    """
    Creates a tiled coarse level.
    Args:
        a: fine-level operator (stiffness matrix).
        b: fine-level mass matrix.
        r: aggregate coarsening.
        p: aggregate interpolation.
        use_r_as_restriction: use R*A*P as coarse operator, if True, else P^T*A*P.

    Returns: coarse level object.
    """
    num_aggregates = a.shape[0] // r.asarray().shape[1]
    r_csr = r.tile(num_aggregates)
    p_csr = p.tile(num_aggregates)
    # Form the SYMMETRIC Galerkin coarse-level operator.
    pt = r_csr if use_r_as_restriction else p_csr.transpose()
    ac = (pt.dot(a)).dot(p_csr)
    bc = (pt.dot(b)).dot(p_csr)
    relaxer = hm.solve.relax.KaczmarzRelaxer(ac, bc)
    return multilevel.Level(ac, bc, relaxer, r, p, r_csr, p_csr)


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
