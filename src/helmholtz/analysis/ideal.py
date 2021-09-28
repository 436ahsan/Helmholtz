import helmholtz as hm
import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
from scipy.optimize import fsolve


def ideal_tv(a, num_examples):
    """Returns a test matrix of the 'num_examples' lowest eigenvectors of a and an array of the
    corresponding eigenvalues.

    Note: scipy may return complex eigenvectors, but a symmetric matrix always has orthogonal real
    eigenvectors. See https://math.stackexchange.com/questions/47764/can-a-real-symmetric-matrix-have-complex-eigenvectors
    for explanation. 𝐴(𝑎+𝑖𝑏)=𝜆(𝑎+𝑖𝑏)⇒𝐴𝑎=𝜆𝑎  and 𝐴𝑏=𝜆𝑏. Thus we return the real part of v here."""
    lam, v = eig(a.todense())
    lam = np.real(lam)
    v = np.real(v)
    ind = np.argsort(np.abs(lam))
    lam = lam[ind]
    v = v[:, ind]
    return np.array(v[:, :num_examples]), lam


def find_singular_kh(discretization, n):
    """
    Finds a k near 0.5 for which the smallest eigenvalue of the discrete Helmholtz operator is singular. That is,
    the smallest eigenvector's wavelength evenly divides the domain size. Here h = 1, so k = kh.

    Uses a sparse eigensolver.

    Args:
        discretization: discretization type ("3-point"|"5-point").
        n: domain size.

    Returns: kh, minimum eigenvalue.
    """
    def func(kh):
        return np.min(np.abs(eigs(hm.linalg.helmholtz_1d_discrete_operator(kh, discretization, n), 1, which="SM")[0]))

    wavelength = 2 * np.pi * n
    root = fsolve(func, wavelength / (2 * np.round(wavelength)))
    return root[0], func(root)