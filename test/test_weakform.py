import numpy as np
import sympy as sp

from lys_fem.mf import weakform, coef, mfem
from lys_fem.mf.weakform import x,y,z,t,dS,dV,grad
from .base import FEMTestCase


class weakform_test(FEMTestCase):

    def test_coef(self):
        mesh = self.generateSimpleMesh(1)
        c = coef.generateCoefficient({1: 1, "default": 0}, mesh.SpaceDimension())
        self.assertTrue(isinstance(c, coef.ScalarCoef))

        c = coef.generateCoefficient({1: 1}, mesh.SpaceDimension())
        self.assertTrue(isinstance(c, coef.ScalarCoef))

        c = coef.generateCoefficient(1, mesh.SpaceDimension())
        self.assertTrue(isinstance(c, coef.ScalarCoef))

        c = coef.generateCoefficient({1: [1,2,3], "default": [1,2,3]}, mesh.SpaceDimension())
        self.assertTrue(isinstance(c, coef.VectorCoef))

        c = coef.generateCoefficient({1: np.eye(3), "default": np.eye(3)}, mesh.SpaceDimension())
        self.assertTrue(isinstance(c, coef.MatrixCoef))

    def test_bilinear(self):
        # linear term
        mesh = self.generateSimpleMesh(3)
        u = weakform.TrialFunction("u", mesh, {0:[], 1:[], 2:[]}, coef.generateCoefficient([1,1,1], mesh.SpaceDimension()), nvar=3)
        v = weakform.TestFunction(u)
        gu, gv = grad(u), grad(v)

        c = sp.Symbol("c")
        cv = sp.Matrix(sp.symbols("cv1, cv2 cv3"))
        m = sp.Matrix([sp.symbols("m11, m12 m13"),sp.symbols("m21, m22 m23"),sp.symbols("m31, m32 m33")])
        wf = c*u.dot(v) + (gu*cv).dot(v) + u.dot(gv.T*cv) + sp.tensorcontraction(sp.tensorcontraction(sp.tensorproduct(gu, m*gv), (0,2)), (0,1))

        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[0], v[0])[0], c)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[0], v[1])[0], 0)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[0], v[2])[0], 0)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[1], v[0])[0], 0)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[1], v[1])[0], c)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[1], v[2])[0], 0)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[2], v[0])[0], 0)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[2], v[1])[0], 0)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[2], v[2])[0], c)

        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[0], v[0])[1], cv)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[1], v[1])[1], cv)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[2], v[2])[1], cv)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[1], v[0])[1], sp.Matrix([0,0,0]))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[2], v[0])[1], sp.Matrix([0,0,0]))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[2], v[1])[1], sp.Matrix([0,0,0]))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[0], v[1])[1], sp.Matrix([0,0,0]))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[0], v[2])[1], sp.Matrix([0,0,0]))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[1], v[2])[1], sp.Matrix([0,0,0]))

        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[0], v[0])[2], sp.Matrix([cv[0],0,0]))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[0], v[1])[2], sp.Matrix([cv[1],0,0]))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[0], v[2])[2], sp.Matrix([cv[2],0,0]))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[1], v[0])[2], sp.Matrix([0,cv[0],0]))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[1], v[1])[2], sp.Matrix([0,cv[1],0]))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[1], v[2])[2], sp.Matrix([0,cv[2],0]))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[2], v[0])[2], sp.Matrix([0,0,cv[0]]))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[2], v[1])[2], sp.Matrix([0,0,cv[1]]))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[2], v[2])[2], sp.Matrix([0,0,cv[2]]))
        
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[0], v[0])[3], m[0,0]*sp.eye(3))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[0], v[1])[3], m[0,1]*sp.eye(3))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[0], v[2])[3], m[0,2]*sp.eye(3))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[1], v[0])[3], m[1,0]*sp.eye(3))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[1], v[1])[3], m[1,1]*sp.eye(3))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[1], v[2])[3], m[1,2]*sp.eye(3))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[2], v[0])[3], m[2,0]*sp.eye(3))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[2], v[1])[3], m[2,1]*sp.eye(3))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u[2], v[2])[3], m[2,2]*sp.eye(3))

        mesh = self.generateSimpleMesh(3)
        u = weakform.TrialFunction("u", mesh, {0:[]}, coef.generateCoefficient(1, mesh.SpaceDimension()))
        v = weakform.TestFunction(u)
        gu, gv = grad(u), grad(v)

        # nonlinear term
        c = sp.Symbol("c")
        cv = sp.Matrix(sp.symbols("cv1, cv2 cv3"))
        m = sp.Matrix([sp.symbols("m11, m12 m13"),sp.symbols("m21, m22 m23"),sp.symbols("m31, m32 m33")])
        wf = gu.dot(cv)*u*v + gu[0]*gu[1]*gu[2]*v

        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u, v)[0], gu.dot(cv))
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, u, v)[1], sp.Matrix([gu[1]*gu[2], gu[2]*gu[0], gu[0]*gu[1]])/3)

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