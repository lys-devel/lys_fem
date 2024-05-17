from lys_fem import ngs
from ..models import magnetostatistics_test

class ngs_em_test(magnetostatistics_test):
    def test_2d_dirichlet(self):
        self.dirichlet_2d(ngs)
