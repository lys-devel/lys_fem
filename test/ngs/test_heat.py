from lys_fem import ngs
from ..models import heat_test

class ngs_heat_test(heat_test):
    def test_1d_dirichlet(self):
        self.dirichlet_1d(ngs)

    def test_1d_neumann(self):
        self.neumann_1d(ngs)

    def test_1d_tdep(self):
        self.tdep_1d(ngs)

    def test_forwardEuler(self):
        self.tdep_forward(ngs)

    def test_bdf2(self):
        self.tdep_bdf2(ngs)
