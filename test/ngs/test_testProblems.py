from lys_fem import ngs
from ..models import testProblems_test

class ngs_testProblems(testProblems_test):
    def test_linear(self):
        self.linear(ngs)

    def test_nonlinear(self):
        self.nonlinear(ngs)

    def test_twoval(self):
        self.twoVars1(ngs)

    def test_twoval_step(self):
        self.twoVars_step(ngs)