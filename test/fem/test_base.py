import sympy as sp
import numpy as np
from lys_fem.fem.base import FEMCoefficient

from ..base import FEMTestCase

class parameters_test(FEMTestCase):
    def test_coef(self):
        # scalar
        c = FEMCoefficient({1: 3, 2: 4})
        self.assertEqual(c[1], 3)
        self.assertEqual(c[2], 4)

        # vector
        c = FEMCoefficient({1: [1,2,3], 2: (4,5,6)})
        self.assertEqual(c[1], [1,2,3])
        self.assertEqual(c[2], [4,5,6])

        # matrix
        c = FEMCoefficient({1: np.eye(3), 2: np.eye(3)*2})
        self.assert_array_equal(c[1], np.eye(3))
        self.assert_array_equal(c[2], np.eye(3)*2)

        # scale
        c = FEMCoefficient({1: [1,2,3]}, scale=10)
        self.assertEqual(c[1], [0.1, 0.2, 0.3])

        # xyz coords
        x,y,z = sp.symbols("x,y,z")
        xs,ys,zs = sp.symbols("x_scaled,y_scaled,z_scaled")
        c = FEMCoefficient({1: [x,y,z]})
        self.assertEqual(c[1], [xs,ys,zs])

        c = FEMCoefficient({1: [x,y,z]}, scale=10)
        self.assertEqual(c[1], [xs/10,ys/10,zs/10])

        # xscale
        c = FEMCoefficient({1: [x+1,y,z]}, scale=10, xscale=2)
        self.assertEqual(c[1], [(xs*2+1)/10,ys/5,zs/5])

        # parameter
        a = sp.Symbol("a")
        c = FEMCoefficient({1: [x+a,y,z]}, scale=10, xscale=2, vars={"a": 2})
        self.assertEqual(c[1], [(xs*2+2)/10,ys/5,zs/5])
