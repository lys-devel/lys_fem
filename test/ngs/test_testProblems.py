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

    def test_twoval_fix(self):
        self.twoVars_fix(ngs)

    def test_consts(self):
        self.consts(ngs)

    def test_fields(self):
        self.fields(ngs)

    def test_loadInitial(self):
        self.loadInitial_1d(ngs)

    def test_tdepField(self):
        self.tdepField(ngs)

    def test_scale(self):
        self.scale(ngs)

    def test_twoval_grad(self):
        self.twoVars_grad(ngs)

    def test_solvers(self):
        self.solver(ngs, "pardiso", None)
        self.solver(ngs, "sparsecholesky", None)
        self.solver(ngs, "CG", "local")
        self.solver(ngs, "CG", "direct")
        self.solver(ngs, "CG", "h1amg")
        self.solver(ngs, "CG", "bddc")
        self.solver(ngs, "GMRES", "local")