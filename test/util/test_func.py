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

    def test_einsum(self):
        fes, u = self._fes3d()
        p = [0.1,0.2,0.3]
        C = util.eval(np.ones((3,3,3,3)))
        v = util.eval(2*np.ones((3,)))
        res = np.einsum("j,ijkl->ijkl",2*np.ones((3,)), np.ones((3,3,3,3)))

        f = util.einsum("j,ijkl->ijkl", v, C)
        self.assertEqual(f.shape, (3,3,3,3))
        self.assertEqual(f.eval(fes).shape, (3,3,3,3))
        self.assertTrue(f.valid)
        self.assertFalse(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertFalse(f.isTimeDependent)
        assert_array_almost_equal(f(fes, p), res)
        #assert_array_almost_equal(util.grad(f)(fes, p), np.zeros((3,3,3,3)))
        assert_array_almost_equal(f.replace({})(fes,p), res)
        assert_array_almost_equal(f.rhs(fes,p), res)
        self.assertEqual(f.lhs(fes,p), 0)
        self.assertFalse(u in f)
