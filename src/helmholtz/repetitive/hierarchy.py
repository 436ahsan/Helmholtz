"""Utilities: multilevel hierarchy for repetitive problems."""
import numpy as np
import helmholtz.setup.multilevel


def create_tiled_coarse_level(a, b, r: np.ndarray, p: np.ndarray) -> helmholtz.setup.multilevel.Level:
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
    # Form Galerkin coarse-level operator.
    ac = (r_csr.dot(a)).dot(p_csr)
    bc = (r_csr.dot(b)).dot(p_csr)
    return helmholtz.setup.multilevel.Level(ac, bc, r, p, r_csr, p_csr)
