import sympy as sp
from .base import FEMTestCase
from lys_fem.mf import mfem
from lys_fem.mf.weakform import TrialFunction, TestFunction, x, y, z, t, dV, dS, grad, sympyCoeff, SubsCoeff

class sympy_test(FEMTestCase):
    def test_funcs(self):
        mesh = dummy(3)
        c = sp.Symbol("c1")
        cv = sp.Matrix(sp.symbols("v1, v2, v3"))
        cm = sp.Matrix([sp.symbols("m11,m12,m13"),sp.symbols("m21,m22,m23"),sp.symbols("m31,m32,m33")])

        u = TrialFunction("u", mesh, None)
        v = TestFunction(u)
        Du, Dv = grad(u), grad(v)
        wf = c*u*v + cv.dot(Du)*v + Du.dot(cm*Dv)

        c1 = sympyCoeff(wf, u, v)
        v1 = sympyCoeff(wf, u, v, True)
        m1 = sympyCoeff(wf, u, v, True, True)
        self.assertEqual(c1, c)
        self.assertEqual(v1, cv)
        self.assertEqual(m1, cm)

        coeffs = {"c1": 1, "v1": 1,"v2": 2, "v3": 3, "m11": 11, "m12": 12, "m13": 13,  "m21": 21, "m22": 22, "m23": 23, "m31": 31, "m32": 32, "m33": 33}
        coeffs = {key: mfem.ConstantCoefficient(value) for key, value in coeffs.items()}

        SubsCoeff(c1, coeffs)
        SubsCoeff(v1, coeffs)
        SubsCoeff(m1, coeffs)

class dummy:
    def __init__(self, dim):
        self._dim=dim
    def Dimension(self):
        return self._dim
