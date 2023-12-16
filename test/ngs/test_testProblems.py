from lys_fem import ngs
from ..models import testProblems_test

class ngs_testProblems(testProblems_test):
    def test_linear(self):
        self.linear(ngs)


