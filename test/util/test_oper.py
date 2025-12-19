import numpy as np
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

    def test_NGSFunction_dot(self):
        fes, u = self._fes3d()
        p = [0.1,0.2,0.3]

        f = util.NGSFunction([util.x, util.y, util.z]).dot(util.eval([1,2,3]))
        self.assertEqual(f.shape, ())
        self.assertTrue(f.valid)
        self.assertFalse(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertFalse(f.isTimeDependent)
        self.assertAlmostEqual(f(fes, p), 1.4)
        assert_array_almost_equal(util.grad(f)(fes, p), [1,2,3])
        self.assertEqual(f.replace({})(fes,p), 1.4)
        self.assertEqual(f.rhs(fes,p), 1.4)
        self.assertEqual(f.lhs(fes,p), 0)
        self.assertFalse(u in f)

    def test_index(self):
        fes, u = self._fes3d()
        p = [0.1,0.2,0.3]

        f = util.NGSFunction([util.x, util.y, util.z])[0]
        self.assertEqual(f.shape, ())
        self.assertTrue(f.valid)
        self.assertFalse(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertFalse(f.isTimeDependent)
        self.assertAlmostEqual(f(fes, p), 0.1)
        assert_array_almost_equal(util.grad(f)(fes, p), [1,0,0])
        self.assertAlmostEqual(f.replace({})(fes,p), 0.1)
        self.assertAlmostEqual(f.rhs(fes,p), 0.1)
        self.assertEqual(f.lhs(fes,p), 0)
        self.assertFalse(u in f)

    def test_index_matrix(self):
        fes, u = self._fes3d()
        p = [0.1,0.2,0.3]

        f = util.NGSFunction([[util.x, 0, 0], [0, util.y, 0], [0, 0, util.z]])[0]
        self.assertEqual(f.shape, (3,))
        self.assertTrue(f.valid)
        self.assertFalse(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertFalse(f.isTimeDependent)
        assert_array_almost_equal(f(fes, p), [0.1, 0, 0])
        assert_array_almost_equal(util.grad(f)(fes, p), [[1,0,0], [0,0,0], [0,0,0]])
        assert_array_almost_equal(f.replace({})(fes,p), [0.1, 0, 0])
        assert_array_almost_equal(f.rhs(fes,p), [0.1, 0, 0])
        self.assertEqual(f.lhs(fes,p), 0)
        self.assertFalse(u in f)
