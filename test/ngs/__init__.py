from scipy import special
import numpy as np
import sympy as sp


from lys_fem import geometry, ngs

from ..base import FEMTestCase

class coef_test(FEMTestCase):
    def test_coef(self):
        mesh = self.generateSimpleNGSMesh(1)
        c = ngs.util.generateCoefficient(1)
        print(type(c))
