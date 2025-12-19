import numpy as np
from numpy.testing import assert_array_almost_equal
from lys_fem import geometry
from lys_fem import util

from ..base import FEMTestCase

class test_coef(FEMTestCase):
    def _fes3d(self, refine=0, size=1, isScalar=True):
        mesh = geometry.GmshMesh([geometry.Box(0,0,0,1,1,1)])
        m = util.Mesh(mesh)

        fs = util.H1("H1", size=size, isScalar=isScalar)
        u, v = fs.trial, fs.test
        wf = util.grad(u).dot(util.grad(v))*util.dx
        return util.FiniteElementSpace(fs, m), u

    def test_trial_scalar(self):
        fes, u = self._fes3d()
        p = [0.1,0.2,0.3]

        f = util.TrialFunction(fes.variables[0])
        self.assertEqual(f.shape, ())
        self.assertEqual(f.eval(fes).shape, ())
        self.assertEqual(util.grad(f).eval(fes).shape, (3,))
        self.assertTrue(f.valid)
        self.assertTrue(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertFalse(f.isTimeDependent)
        self.assertEqual(f.rhs(fes, p), 0)
        self.assertEqual(f.lhs, f)
        self.assertTrue(u in f)

    def test_trial_vector(self):
        fes, u = self._fes3d(size=2)
        p = [0.1,0.2,0.3]

        f = util.TrialFunction(fes.variables[0])
        self.assertEqual(f.shape, (3,))
        self.assertEqual(f.eval(fes).shape, (3,))
        self.assertEqual(util.grad(f).eval(fes).shape, (3,3))
        self.assertTrue(f.valid)
        self.assertTrue(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertFalse(f.isTimeDependent)
        self.assertEqual(f.rhs(fes, p), 0)
        self.assertEqual(f.lhs, f)
        self.assertTrue(u in f)

    def test_trial_matrix(self):
        fes, u = self._fes3d(size = (1,1))
        p = [0.1,0.2,0.3]

        f = util.TrialFunction(fes.variables[0])
        self.assertEqual(f.shape, (3,3))
        self.assertEqual(f.eval(fes).shape, (3,3))
        self.assertEqual(util.grad(f).eval(fes).shape, (3,3,3))
        self.assertTrue(f.valid)
        self.assertTrue(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertFalse(f.isTimeDependent)
        self.assertEqual(f.rhs(fes, p), 0)
        self.assertEqual(f.lhs, f)
        self.assertTrue(u in f)
