from lys_fem import ngs
from ..models import testProblems_test

class test(testProblems_test):
    def test_linear(self):
        self.linear(ngs)

    def test_nonlinear(self):
        self.nonlinear(ngs)

    def test_twoval(self):
        self.twoVars1(ngs)

    def test_twoval_step(self):
        self.twoVars_step(ngs)

    def test_consts(self):
        self.consts(ngs)

    def test_fields(self):
        self.fields(ngs)

    def test_loadInitial(self):
        self.loadInitial_1d(ngs)
