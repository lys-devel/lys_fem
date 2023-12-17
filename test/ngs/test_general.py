from lys_fem import ngs
from ..models import poisson_test

class ngs_heat_test(poisson_test):
    def test_3d_dirichlet(self):
        self.dirichlet_3d(ngs)



