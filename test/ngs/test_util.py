from netgen.geom2d import unit_square


from lys_fem import geometry
from lys_fem.fem import FEMProject
from lys_fem.ngs import util, mesh

from ..base import FEMTestCase

class test_util(FEMTestCase):
    def _make_mesh(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        return mesh.generateMesh(p)
    
    def test_mpi(self):
        self.assertFalse(util.mpi.isParallel())

    def test_space(self):
        m = self._make_mesh()

        obj = util.H1("u", size = 1, order = 1, isScalar=True)
        self.assertEqual(obj.size, 1)
        self.assertTrue(obj.isScalar)

        h1 = obj.eval(m)
        self.assertEqual(sum(h1.FreeDofs()), m.nodes)

        h1 = util.H1("u", size = 1, order = 2).eval(m)
        self.assertEqual(sum(h1.FreeDofs()), m.nodes*2-1)

        h1 = util.H1("u", size = 1, order = 1, dirichlet=["boundary1"]).eval(m)
        self.assertEqual(sum(h1.FreeDofs()), m.nodes-1)

        h1 = util.H1("u", size = 1, order = 1, definedon="domain1").eval(m)
        self.assertEqual(sum(h1.FreeDofs()), m.nodes//2+1)

        h1 = util.H1("u", size = 2, order = 1).eval(m)
        self.assertEqual(sum(h1.FreeDofs()), m.nodes*2)

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

