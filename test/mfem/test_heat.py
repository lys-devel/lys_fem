from lys_fem import mf
from ..models import heat_test

class mfem_heat_test(heat_test):
    def test_1d_dirichlet(self):
        self.dirichlet_1d(mf)

    def test_1d_neumann(self):
        self.neumann_1d(mf)

    def test_1d_tdep(self):
        self.tdep_1d(mf)


