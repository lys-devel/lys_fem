import numpy as np
import sympy as sp

from lys_fem.mf import weakform, mfem
from lys_fem.mf.weakform import x,y,z,t,dS,dV,grad
from ..base import FEMTestCase


class weakform_test(FEMTestCase):

    def test_coef(self):
        mesh = self.generateSimpleMesh(1)
        c1 = mfem.generateCoefficient({1: 1, "default": 0})
        self.assertTrue(isinstance(c1, mfem.SympyCoefficient))

        c2 = mfem.generateCoefficient({1: 1})
        self.assertTrue(isinstance(c2, mfem.ConstantCoefficient))

        c3 = mfem.generateCoefficient(1)
        self.assertTrue(isinstance(c3, mfem.ConstantCoefficient))

        c4 = mfem.generateCoefficient({1: [1,2,3], "default": [1,2,3]})
        self.assertTrue(isinstance(c4, mfem.VectorArrayCoefficient))

        c5 = mfem.generateCoefficient({1: np.eye(3), "default": np.eye(3)})
        self.assertTrue(isinstance(c5, mfem.MatrixArrayCoefficient))

        m = mfem.MatrixArrayCoefficient([[c1, c2], [c3, c2]])

    def test_bilinear(self):
        # linear term
        mesh = self.generateSimpleMesh(3)
        u = weakform.TrialFunction("u", mesh, {0:[], 1:[], 2:[]}, mfem.generateCoefficient([1,1,1]), nvar=3)
        v = weakform.TestFunction(u)
        gu, gv = grad(u), grad(v)

        c = sp.Symbol("c")
        cv = sp.Matrix(sp.symbols("cv1, cv2 cv3"))
        m = sp.Matrix([sp.symbols("m11, m12 m13"),sp.symbols("m21, m22 m23"),sp.symbols("m31, m32 m33")])
        wf = c*u.dot(v) + (gu*cv).dot(v) + u.dot(gv.T*cv) + sp.tensorcontraction(sp.tensorcontraction(sp.tensorproduct(gu, m*gv), (0,2)), (0,1))
        vars = [uu for uu in u]

        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[0], v[0])[0], c)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[0], v[1])[0], 0)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[0], v[2])[0], 0)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[1], v[0])[0], 0)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[1], v[1])[0], c)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[1], v[2])[0], 0)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[2], v[0])[0], 0)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[2], v[1])[0], 0)
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[2], v[2])[0], c)

        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[0], v[0])[1], [cv[0], cv[1], cv[2]])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[1], v[1])[1], [cv[0], cv[1], cv[2]])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[2], v[2])[1], [cv[0], cv[1], cv[2]])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[1], v[0])[1], [0,0,0])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[2], v[0])[1], [0,0,0])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[2], v[1])[1], [0,0,0])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[0], v[1])[1], [0,0,0])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[0], v[2])[1], [0,0,0])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[1], v[2])[1], [0,0,0])

        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[0], v[0])[2], [cv[0],0,0])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[0], v[1])[2], [cv[1],0,0])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[0], v[2])[2], [cv[2],0,0])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[1], v[0])[2], [0,cv[0],0])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[1], v[1])[2], [0,cv[1],0])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[1], v[2])[2], [0,cv[2],0])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[2], v[0])[2], [0,0,cv[0]])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[2], v[1])[2], [0,0,cv[1]])
        self.assertEqual(weakform._BilinearForm._getCoeffs(wf, vars, u[2], v[2])[2], [0,0,cv[2]])
     
        self.assertEqual(sp.Matrix(weakform._BilinearForm._getCoeffs(wf, vars, u[0], v[0])[3]), m[0,0]*sp.eye(3))
        self.assertEqual(sp.Matrix(weakform._BilinearForm._getCoeffs(wf, vars, u[0], v[1])[3]), m[0,1]*sp.eye(3))
        self.assertEqual(sp.Matrix(weakform._BilinearForm._getCoeffs(wf, vars, u[0], v[2])[3]), m[0,2]*sp.eye(3))
        self.assertEqual(sp.Matrix(weakform._BilinearForm._getCoeffs(wf, vars, u[1], v[0])[3]), m[1,0]*sp.eye(3))
        self.assertEqual(sp.Matrix(weakform._BilinearForm._getCoeffs(wf, vars, u[1], v[1])[3]), m[1,1]*sp.eye(3))
        self.assertEqual(sp.Matrix(weakform._BilinearForm._getCoeffs(wf, vars, u[1], v[2])[3]), m[1,2]*sp.eye(3))
        self.assertEqual(sp.Matrix(weakform._BilinearForm._getCoeffs(wf, vars, u[2], v[0])[3]), m[2,0]*sp.eye(3))
        self.assertEqual(sp.Matrix(weakform._BilinearForm._getCoeffs(wf, vars, u[2], v[1])[3]), m[2,1]*sp.eye(3))
        self.assertEqual(sp.Matrix(weakform._BilinearForm._getCoeffs(wf, vars, u[2], v[2])[3]), m[2,2]*sp.eye(3))

        mesh = self.generateSimpleMesh(3)
        u = weakform.TrialFunction("u", mesh, {0:[]}, mfem.generateCoefficient(1))
        v = weakform.TestFunction(u)
        gu, gv = grad(u), grad(v)

        # nonlinear term
        c = sp.Symbol("c")
        cv = sp.Matrix(sp.symbols("cv1, cv2 cv3"))
        m = sp.Matrix([sp.symbols("m11, m12 m13"),sp.symbols("m21, m22 m23"),sp.symbols("m31, m32 m33")])
        wf = gu.dot(cv)*u*v + gu[0]*gu[1]*gu[2]*v

        vars = [u]

        self.assertEqual(sp.Matrix(weakform._BilinearForm._getCoeffs(wf, vars, u, v)[1]), cv*u + sp.Matrix([gu[1]*gu[2], gu[2]*gu[0], gu[0]*gu[1]])/3)
