from ..models import poisson_test

ngs = 1
class ngs_poisson_test(poisson_test):
    def test_3d_dirichlet(self):
        self.dirichlet_3d(ngs)

    def test_2d_dirichlet(self):
        self.dirichlet_2d(ngs)

    def test_1d_dirichlet(self):
        self.dirichlet_1d(ngs)

    def test_3d_infinite(self):
        self.infinite_3d(ngs)