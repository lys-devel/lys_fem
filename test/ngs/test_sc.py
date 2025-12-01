from ..models import semiconductor_test

ngs = 1
class ngs_sc_test(semiconductor_test):
    def test_1d_stationary(self):
        self.stationary(ngs)
