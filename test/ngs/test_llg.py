from lys_fem import ngs
from ..models import LLG_test

class ngs_LLG_test(LLG_test):
    def test_precession(self):
        self.precession(ngs)

    def test_stationary(self):
        self.stationary(ngs)

    def test_anisU(self):
        self.anisU(ngs)

    def test_DW(self):
        self.domainWall(ngs)

    def test_deform(self):
        self.deformation(ngs)

    def test_scalar(self):
        self.scalar(ngs)

    def test_scalar_em(self):
        self.scalar_em(ngs)

    def test_demag_em(self):
        self.demagnetization_em(ngs)