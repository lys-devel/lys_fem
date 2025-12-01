from ..models import magnetostatistics_test

ngs = 1
class ngs_em_test(magnetostatistics_test):
    def test_2d_dirichlet(self):
        self.dirichlet_2d(ngs)

    def test_demag(self):
        self.demagnetization(ngs)