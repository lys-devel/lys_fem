from lys_fem import ngs
from ..models import LLG_test

class ngs_LLG_test(LLG_test):
    def test_precession(self):
        self.precession(ngs)

    def test_stationary(self):
        self.stationary(ngs)

    def test_demagnetization(self):
        self.demagnetization(ngs)