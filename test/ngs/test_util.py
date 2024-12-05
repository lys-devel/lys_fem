import ngsolve
from netgen.geom2d import unit_square

from lys_fem.ngs import util
from ..base import FEMTestCase

class test_util(FEMTestCase):     
    def test_NGSFunction(self):
        mesh = ngsolve.Mesh(unit_square.GenerateMesh(maxh=0.2))
        mp = mesh(0,0)
        util.dimension = 2

        f = util.NGSFunction(1)
        self.assertEqual(f.shape, ())
        self.assertTrue(f.valid)
        self.assertFalse(f.hasTrial)
        self.assertFalse(f.isNonlinear)
        self.assertFalse(f.isTimeDependent)
        self.assertAlmostEqual(f.eval()(mp), 1)
        self.assertEqual(f.grad()(mp), (0,0))
        self.assertAlmostEqual(f.integrate(mesh), 1)
        self.assertEqual(f.replace({}).eval()(mp), 1)
        self.assertEqual(f.rhs.eval()(mp), 1)
        self.assertEqual(f.lhs.eval()(mp), 0)

