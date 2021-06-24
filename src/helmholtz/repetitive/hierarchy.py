"""Utilities: multilevel hierarchy for repetitive problems."""
import numpy as np
import scipy.sparse
import helmholtz as hm
import helmholtz.hierarchy.multilevel as multilevel
import helmholtz.repetitive.hierarchy as hierarchy


def create_tiled_coarse_level(a: scipy.sparse.spmatrix, b: scipy.sparse.spmatrix, r: np.ndarray, p: np.ndarray) -> \
        multilevel.Level:
    """
    Creates a tiled coarse level.
    Args:
        a: fine-level operator (stiffness matrix).
        b: fine-level mass matrix.
        r: aggregate coarsening.
        p: aggregate interpolation.

    Returns: coarse level object.
    """
    num_aggregates = a.shape[0] // r.asarray().shape[1]
    r_csr = r.tile(num_aggregates)
    p_csr = p.tile(num_aggregates)
    # Form the SYMMETRIC Galerkin coarse-level operator.
    pt = p_csr.transpose()
    ac = (pt.dot(a)).dot(p_csr)
    bc = (pt.dot(b)).dot(p_csr)
    relaxer = hm.solve.relax.KaczmarzRelaxer(ac, bc)
    return multilevel.Level(ac, bc, relaxer, r, p, r_csr, p_csr)


def create_finest_level(a: scipy.sparse.spmatrix, relaxer: None) -> multilevel.Level:
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
