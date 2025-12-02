import numpy as np
from netgen.geom2d import unit_square

import ngsolve
from numpy.testing import assert_almost_equal
from lys_fem import geometry
from lys_fem.fem import FEMProject
from lys_fem import util

from ..base import FEMTestCase

class test_util(FEMTestCase):
    def _make_mesh(self, refine=0):
        p = FEMProject(1)
        p.mesher.setRefinement(refine)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        return p.mesh
    
    def test_space(self):
        m = self._make_mesh()

        obj = util.H1("u", size = 1, order = 1, isScalar=True)
        self.assertEqual(obj.size, 1)
        self.assertTrue(obj.isScalar)

        h1 = obj.eval(m)
        self.assertEqual(sum(h1.FreeDofs()), m.nodes)

        h1 = util.H1("u", size = 1, order = 2).eval(m)
        self.assertEqual(sum(h1.FreeDofs()), m.nodes*2-1)

        h1 = util.H1("u", size = 1, order = 1, geometries=[1]).eval(m)
        self.assertEqual(sum(h1.FreeDofs()), m.nodes//2+1)

        h1 = util.H1("u", size = 2, order = 1).eval(m)
        self.assertEqual(sum(h1.FreeDofs()), m.nodes*2)

    def test_grid(self):
        import numpy as np
        p = FEMProject(1)
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        m = p.mesh

        sp = util.H1("u", size = 1, order = 2, isScalar=True)
        fes = util.FiniteElementSpace(sp, m)
        x = util.NGSFunction(ngsolve.x, name="x")

        f = fes.gridFunction([x*x])
        g = ngsolve.grad(f)

        x_list = np.linspace(0,1,100)
        f_calc = [xx**2 for xx in x_list]
        f_eval = [f(m(xx)) for xx in x_list]
        g_calc = [2*xx for xx in x_list]
        g_eval = np.array([g(m(xx)) for xx in x_list])

        assert_almost_equal(f_calc, f_eval)
        assert_almost_equal(g_calc, g_eval[:,0])

    def test_grid_mesh(self):
        # Check if we can project grid function between different mesh.
        m1 = self._make_mesh(refine=0)
        m2 = self._make_mesh(refine=2)

        sp = util.H1("u", size = 1, order = 2, isScalar=True)
        fes1 = util.FiniteElementSpace(sp, m1)
        fes2 = util.FiniteElementSpace(sp, m2)
        x = util.NGSFunction(ngsolve.x, name="x")

        f1 = fes1.gridFunction([x*x])
        f2 = fes2.gridFunction()

        f2.setComponent(sp, util.GridField(f1, sp))
        g2 = util.GridField(f2, sp)

        x_list = np.linspace(0,1)
        res = g2(fes2, x_list)

        assert_almost_equal(res, x_list**2)

    def test_poisson(self):
        m = self._make_mesh()

        # H1 space and weakform of Poisson equation.
        fs = util.H1("H1")
        u, v = fs.trial, fs.test
        wf = util.grad(u).dot(util.grad(v))*util.dx

        # Make finite element space with a mesh, and make grid function on it.
        fes = util.FiniteElementSpace(fs, m)
        g = fes.gridFunction()

        # Solve Poisson equation on given finite element space.
        util.Solver(fes, wf, linear={"solver": "pardiso"}).solve(g)

    def test_NGSFunction(self):
        return
        m = mesh.NGSMesh(unit_square.GenerateMesh(maxh=0.2), None)
        mp = m(0,0)
        util.dimension = 2

        f = util.NGSFunction(1)
        self.assertEqual(f.shape, ())
        self.assertTrue(f.valid)
        self.assertFalse(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertFalse(f.isTimeDependent)
        self.assertAlmostEqual(f.eval(m)(mp), 1)
        self.assertEqual(f.grad(m)(mp), (0,0))
        self.assertEqual(f.replace({}).eval(m)(mp), 1)
        self.assertEqual(f.rhs.eval(m)(mp), 1)
        self.assertEqual(f.lhs.eval(m)(mp), 0)

