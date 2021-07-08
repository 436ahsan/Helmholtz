import helmholtz as hm
import numpy as np
from numpy.linalg import eig


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

