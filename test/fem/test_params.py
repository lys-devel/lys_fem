import sympy as sp
from lys_fem.fem.parameters import Parameters

from ..base import FEMTestCase

class parameters_test(FEMTestCase):
    def test_parameters(self):
        p = Parameters()

        a,b = sp.symbols("a,b")

        p[a] = 2
        p[b] = a**2 

        sol = p.getSolved()
        self.assertEqual(sol[a], 2)
        self.assertEqual(sol[b], 4)
