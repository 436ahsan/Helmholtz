"""Local Fourier Analysis (LFA) of the 1D Helmholtz problem."""
import numpy as np
from numpy import pi


def a_poly(kh, discretization):
    if discretization == "3-point":
        a = np.array([1, -2 + kh ** 2, 1])
    elif discretization == "5-point":
        a = np.array([-1, 16, -30 + 12 * kh ** 2, 16, -1])
    else:
        raise Exception("Unsupported discretization type")
    return np.poly1d(a)


def gs(a, t):
    """Returns the Gauss-Seidel relaxation symbol of an operator 'a'. a must be odd-length, and centered at
    len(a) // 2."""
    mid = len(a) // 2
    left = np.poly1d(a.c[:mid + 1])
    right = np.poly1d(np.concatenate((a.c[mid + 1:][::-1], [0])))
    r = np.abs(right(np.exp(1j * t)) / left(np.exp(-1j * t)))
    return r


def kaczmarz(stencil, t):
    """
    Returns the Kaczmarz relaxation symbol of a symmetric operator.
    Args:
        stencil: a symmetric operator's stencil.
        t: array of scaled frequency (theta) values.

    Returns:
        Kaczmarz relaxation symbol at t.
    """
    # A is symmetric, so A*A' = A^2.
    return gs(np.polymul(stencil, stencil), t)


def a(stencil, t):
    return np.real(np.exp(-1j * t * (len(stencil) // 2)) * stencil(np.exp(1j * t)))


def harmonics(t):
    return t, t + pi


def geometric_interpolation(t, kind):
    if kind == "constant":
        return 0.5 * (1 + np.exp(-1j * t))
    elif kind == "linear":
        return 0.5 * (1 + np.cos(t))
    else:
        raise Exception("Unsupported restriction type {}".format(kind))


def asymptotic_conv_factor(symbol):
    return np.max(np.abs(np.linalg.eig(symbol)[0]))


class LfaComputer:
    def __init__(self, kh=0, discretization="3-point", relax="gs",
                 restriction="constant", interpolation="constant", coarsening="galerkin"):
        self.kh = kh
        self.discretization = discretization
        self.relax = relax
        self.restriction = restriction
        self.interpolation = interpolation
        self.coarsening = coarsening
        self.stencil = a_poly(self.kh, self.discretization)

    def clc(self, t):
        A = np.diag([a(self.stencil, th) for th in harmonics(t)])
        R = np.matrix([[geometric_interpolation(th, self.restriction) for th in harmonics(t)]])
        if self.interpolation == "restriction_adjoint":
            P = R.H
        else:
            P = np.matrix([[geometric_interpolation(th, self.interpolation) for th in harmonics(t)]]).H
        AC = self._coarse_operator(t, A, R, P)
        return np.eye(2) - P.dot(np.linalg.solve(AC, R.dot(A)))

    def two_level_symbol(self, t, nu):
        relax = self._relax(t)
        return self.clc(t).dot(np.linalg.matrix_power(relax, nu))

    @property
    def theta(self):
        return np.linspace(-pi, pi, 101) + 1e-5

    def two_level_aymptotic_amplification(self, nu):
        return np.array([asymptotic_conv_factor(self.two_level_symbol(ti, nu)) for ti in self.theta])

    def _relax(self, t):
        if self.relax == "gs":
            return np.diag([gs(self.stencil, th) for th in harmonics(t)])
        elif self.relax == "relax":
            return np.diag([kaczmarz(self.stencil, th) for th in harmonics(t)])
        else:
            raise Exception("Unsupported relaxation type {}".format(self.relax))

    def _coarse_operator(self, t, A, R, P):
        if self.coarsening == "galerkin":
            return (R.dot(A)).dot(P)
        elif self.coarsening == "direct":
            return np.array([[0.25 * a(self.stencil, 2 * t)]])
        else:
            raise Exception("Unsupported coarsening type {}".format(self.coarsening))
