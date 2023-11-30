import numpy as np
import sympy as sp

from lys_fem.mf import weakform, coef, mfem
from lys_fem.mf.weakform import x,y,z,t,dS,dV
from .base import FEMTestCase


class weakform_test(FEMTestCase):
    def test_coef(self):
        mesh = self.generateSimpleMesh(1)
        c = coef.generateCoefficient({1: 1, "default": 0}, mesh)
        self.assertTrue(isinstance(c, coef.ScalarCoef))

        c = coef.generateCoefficient({1: 1}, mesh)
        self.assertTrue(isinstance(c, coef.ScalarCoef))

        c = coef.generateCoefficient(1, mesh)
        self.assertTrue(isinstance(c, coef.ScalarCoef))

        c = coef.generateCoefficient({1: [1,2,3], "default": [1,2,3]}, mesh)
        self.assertTrue(isinstance(c, coef.VectorCoef))

        c = coef.generateCoefficient({1: np.eye(3), "default": np.eye(3)}, mesh)
        self.assertTrue(isinstance(c, coef.MatrixCoef))

    def test_trial_1d(self):
        mesh = self.generateSimpleMesh(1)
        u = weakform.TrialFunction("u", mesh, [1], coef.generateCoefficient(1, mesh))
        v = weakform.TestFunction(u)

        c = sp.Symbol("c")
        c_val = coef.generateCoefficient(2, mesh)
        wf = c*v*dV

        b = weakform._LinearForm.getVector(wf, v, {"c": c_val})
        bp = u.mfem.dualToPrime(b)
        self.assert_array_almost_equal([bb for bb in bp], 2)

    def test_trial_2d(self):
        mesh = self.generateSimpleMesh(2)
        u = weakform.TrialFunction("u", mesh, {0: [4,6], 1:[]}, coef.generateCoefficient(sp.Matrix([1,2]), mesh), nvar=2)
        v = weakform.TestFunction(u)

        c = sp.Symbol("c")
        c_val = coef.generateCoefficient(2, mesh)
        wf = c*u[0]*v[1]*dV

        x1 = u[0].mfem.x
        x2 = u[1].mfem.x

        b1 = mfem.Vector(x1)
        b2 = mfem.Vector([0] * x2.Size())

        K1 = weakform._BilinearForm.getMatrix(wf, u[0], v[0], {"c": c_val})
        K2 = weakform._BilinearForm.getMatrix(wf, u[0], v[1], {"c": c_val}, x1, b2)

        v = mfem.Vector([1]*K2.Width())
        v2 = mfem.Vector([0]*K2.Width())

        K2.Mult(v, v2)

        print([x for x in v2])