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

    def test_save(self):
        p = Parameters()
        a,b,c = sp.symbols("a,b,c")

        p[a] = 2
        p[b] = a**2 
        p[c] = sp.sin(a)

        d = p.saveAsDictionary()
        p2 = Parameters.loadFromDictionary(d)
        self.assertEqual(p2[a], 2)
        self.assertEqual(p2[b], a**2)
        self.assertEqual(p2[c], sp.sin(a))