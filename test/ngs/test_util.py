from netgen.geom2d import unit_square

from lys_fem.ngs import util, mesh
from ..base import FEMTestCase

class test_util(FEMTestCase):     
    def test_NGSFunction(self):
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

