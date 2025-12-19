import numpy as np

from numpy.testing import assert_almost_equal
from lys_fem import geometry
from lys_fem import util

from ..base import FEMTestCase

class test_util(FEMTestCase):
    def _make_mesh(self, refine=0):
        mesh = geometry.GmshMesh([geometry.Line(0,0,0,1,0,0), geometry.Line(1,0,0,2,0,0)], refine=refine)
        return util.Mesh(mesh)

    def _fes3d(self, refine=0, size=1, isScalar=True):
        mesh = geometry.GmshMesh([geometry.Box(0,0,0,1,1,1)])
        m = util.Mesh(mesh)

        fs = util.H1("H1", size=size, isScalar=isScalar)
        u, v = fs.trial, fs.test
        wf = util.grad(u).dot(util.grad(v))*util.dx
        return util.FiniteElementSpace(fs, m), u

    def test_grid_scalar(self):
        fes, u = self._fes3d()
        p = [0.1,0.2,0.3]

        g = fes.gridFunction(util.eval("x"))

        f = util.GridField(g, fes.variables[0])
        self.assertEqual(f.shape, ())
        self.assertEqual(f.eval(fes).shape, ())
        self.assertEqual(util.grad(f).shape, (3,))
        self.assertEqual(util.grad(f).eval(fes).shape, (3,))
        self.assertAlmostEqual(f(fes, p), 0.1)
        self.assert_array_almost_equal(util.grad(f)(fes, p), [1,0,0])
        self.assertTrue(f.valid)
        self.assertFalse(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertTrue(f.isTimeDependent)
        self.assertAlmostEqual(f.rhs(fes, p), 0.1)
        self.assertAlmostEqual(f.rhs(fes, p), 0.1)
        self.assertEqual(f.lhs(fes, p), 0)
        self.assertFalse(u in f)

    def test_grid_vector(self):
        fes, u = self._fes3d(size=2)
        p = [0.1,0.2,0.3]

        g = fes.gridFunction(util.eval(["x", "2*y"]))

        f = util.GridField(g, fes.variables[0])
        self.assertEqual(f.shape, (3,))
        self.assertEqual(f.eval(fes).shape, (3,))
        self.assertEqual(util.grad(f).shape, (3,3))
        self.assertEqual(util.grad(f).eval(fes).shape, (3,3))
        self.assert_array_almost_equal(f(fes, p), [0.1, 0.4, 0])
        self.assert_array_almost_equal(util.grad(f)(fes, p), [[1,0,0], [0,2,0], [0,0,0]])
        self.assertTrue(f.valid)
        self.assertFalse(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertTrue(f.isTimeDependent)
        self.assert_array_almost_equal(f.rhs(fes, p), [0.1, 0.4, 0])
        self.assert_array_almost_equal(f.rhs(fes, p), [0.1, 0.4, 0])
        self.assert_array_almost_equal(f.lhs(fes, p), [0, 0, 0])
        self.assertFalse(u in f)

    def test_grid_matrix(self):
        fes, u = self._fes3d(size = (2,2))
        p = [0.1,0.2,0.3]

        g = fes.gridFunction(util.eval([["x", "2*y"], ["3*z",4]]))
        res = [[0.1, 0.4, 0], [0.9, 4, 0], [0, 0, 0]]

        f = util.GridField(g, fes.variables[0])
        self.assertEqual(f.shape, (3,3))
        self.assertEqual(f.eval(fes).shape, (3,3))
        self.assertEqual(util.grad(f).shape, (3,3,3))
        self.assertEqual(util.grad(f).eval(fes).shape, (3,3,3))
        self.assert_array_almost_equal(f(fes, p), res)
        self.assert_array_almost_equal(util.grad(f)(fes, p), [[[1,0,0], [0,0,0], [0,0,0]], [[0,2,0], [0,0,0], [0,0,0]], [[0,0,0], [3,0,0], [0,0,0]]])
        self.assertTrue(f.valid)
        self.assertFalse(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertTrue(f.isTimeDependent)
        self.assert_array_almost_equal(f.rhs(fes, p), res)
        self.assert_array_almost_equal(f.rhs(fes, p), res)
        self.assert_array_almost_equal(f.lhs(fes, p), np.zeros((3,3)))
        self.assertFalse(u in f)
    
    def test_error(self):
        m = self._make_mesh()

        fs = util.H1("H1", dirichlet=[[1,3]], order=1, isScalar=True)
        u, v = fs.trial, fs.test
        wf = u * util.grad(u).dot(util.grad(v))*util.dx

        fes = util.FiniteElementSpace(fs, m)
        g = fes.gridFunction([util.x])

        # Solve nonlinear Poisson equation on given finite element space.
        util.Solver(fes, wf, linear={"solver": "pardiso"}).solve(g)
        f = util.GridField(g, fs)
        gf = f.error()

        x_list = np.linspace(0,2,7)
        err = gf(fes, x_list)

        assert_almost_equal(err, [5.71909584e-02, 5.71909584e-02, 4.93297009e-02, 1.16153092e-03, 3.19122990e-04, 1.34545672e-04, 8.54751546e-05])
