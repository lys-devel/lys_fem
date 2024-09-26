from lys_fem import ngs
from ..models import elasticity_test

class ngs_elasticity_test(elasticity_test):
    def test_1d_dirichlet(self):
        self.dirichlet_1d(ngs)

    def test_2d_dirichlet(self):
        self.dirichlet_2d(ngs)

    def test_3d_dirichlet(self):
        self.dirichlet_3d(ngs)

    def test_1d_tdep(self):
        self.tdep_1d(ngs)

    def test_1d_te(self):
        self.thermoelasticity_1d(ngs)

    def test_2d_te(self):
        self.thermoelasticity_2d(ngs)
