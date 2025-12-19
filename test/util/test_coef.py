import numpy as np

import ngsolve
from numpy.testing import assert_array_almost_equal
from lys_fem import geometry
from lys_fem import util

from ..base import FEMTestCase

class test_coef(FEMTestCase):
    def _fes1d(self, refine=0):
        mesh = geometry.GmshMesh([geometry.Line(0,0,0,1,0,0)])
        m = util.Mesh(mesh)
        fs = util.H1("H1")
        u, v = fs.trial, fs.test
        wf = util.grad(u).dot(util.grad(v))*util.dx
        return util.FiniteElementSpace(fs, m), u

    def _fes3d(self, refine=0):
        mesh = geometry.GmshMesh([geometry.Box(0,0,0,1,1,1)])
        m = util.Mesh(mesh)
        fs = util.H1("H1")
        u, v = fs.trial, fs.test
        wf = util.grad(u).dot(util.grad(v))*util.dx
        return util.FiniteElementSpace(fs, m), u

    def test_NGSFunction_const(self):
        fes, u = self._fes1d()

        f = util.NGSFunction(1)
        self.assertEqual(f.shape, ())
        self.assertTrue(f.valid)
        self.assertFalse(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertFalse(f.isTimeDependent)
        self.assertAlmostEqual(f(fes, 0.5), 1)
        assert_array_almost_equal(util.grad(f)(fes, 0.5), [0,0,0])
        self.assertEqual(f.replace({})(fes,0.5), 1)
        self.assertEqual(f.rhs(fes,0.5), 1)
        self.assertEqual(f.lhs(fes,0.5), 0)
        self.assertEqual(str(f), "1")
        self.assertFalse(u in f)

    def test_NGSFunction_x(self):
        fes, u = self._fes1d()

        f = util.x
        self.assertEqual(f.shape, ())
        self.assertTrue(f.valid)
        self.assertFalse(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertFalse(f.isTimeDependent)
        self.assertAlmostEqual(f(fes, 0.5), 0.5)
        assert_array_almost_equal(util.grad(f)(fes, 0.5), [1,0,0])
        self.assertEqual(f.replace({})(fes,0.5), 0.5)
        self.assertEqual(f.rhs(fes,0.5), 0.5)
        self.assertEqual(f.lhs(fes,0.5), 0)
        self.assertEqual(str(f), "x")
        self.assertFalse(u in f)

    def test_NGSFunction_vector(self):
        fes, u = self._fes3d()
        p = [0.5, 0.2, 0.7]

        f = util.NGSFunction([util.x, util.y, 1])
        self.assertEqual(f.shape, (3,))
        self.assertTrue(f.valid)
        self.assertFalse(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertFalse(f.isTimeDependent)
        assert_array_almost_equal(f(fes, p), [0.5,0.2,1])
        assert_array_almost_equal(util.grad(f)(fes, p), [[1,0,0], [0,1,0], [0,0,0]])
        assert_array_almost_equal(f.replace({})(fes,p), [0.5,0.2,1])
        assert_array_almost_equal(f.rhs(fes,p), [0.5,0.2,1])
        assert_array_almost_equal(f.lhs(fes,p), [0,0,0])
        self.assertFalse(u in f)

    def test_NGSFunction_matrix(self):
        fes, u = self._fes3d()
        p = [0.5, 0.2, 0.7]
        value = [[0.5,0.2,1], [1,2,3], [0,1,0.7]]

        f = util.NGSFunction([[util.x, util.y, 1], [1,2,3], [0, 1, util.z]])
        self.assertEqual(f.shape, (3,3))
        self.assertTrue(f.valid)
        self.assertFalse(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertFalse(f.isTimeDependent)
        assert_array_almost_equal(f(fes, p), value)
        assert_array_almost_equal(util.grad(f)(fes, p), [[[1,0,0], [0,0,0], [0,0,0]], [[0,1,0], [0,0,0], [0,0,0]], [[0,0,0], [0,0,0], [0,0,1]]])
        assert_array_almost_equal(f.replace({})(fes,p), value)
        assert_array_almost_equal(f.rhs(fes,p), value)
        assert_array_almost_equal(f.lhs(fes,p), np.array(value)*0)
        self.assertFalse(u in f)
