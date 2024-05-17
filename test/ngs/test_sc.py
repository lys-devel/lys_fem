from lys_fem import ngs
from ..models import semiconductor_test

class ngs_sc_test(semiconductor_test):
    def test_1d_stationary(self):
        self.stationary(ngs)
