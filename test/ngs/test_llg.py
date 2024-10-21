from lys_fem import ngs
from ..models import LLG_test

class ngs_LLG_test(LLG_test):
    def test_precession1(self):
        self.precession(ngs)

    def test_precession2(self):
        self.precession(ngs, constraint="Lagrange")

    def test_precession3(self):
        self.precession(ngs, constraint="Alouges", discretization="Alouges theta")

    def test_anisU(self):
        self.anisU(ngs)

    def test_anisU2(self):
        self.anisU(ngs, constraint="Alouges", discretization="Alouges theta")

    def test_DW(self):
        self.domainWall(ngs)

    def test_DW2(self):
        self.domainWall(ngs, constraint="Alouges", discretization="Alouges theta")

    def test_DW_3d(self):
        return
        self.domainWall_3d(ngs, constraint="Alouges", discretization="Alouges theta")

    def test_deform(self):
        return
        self.deformation(ngs)

    def test_scalar(self):
        self.scalar(ngs)

    def test_scalar_em(self):
        self.scalar_em(ngs)

    def test_demag_em(self):
        self.demagnetization_em(ngs)
    
    def test_damping(self):
        self.damping(ngs)
