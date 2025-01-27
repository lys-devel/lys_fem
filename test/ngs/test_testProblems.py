from lys_fem import ngs
from ..models import testProblems_test

class test(testProblems_test):
    def test_linear(self):
        self.linear(ngs)

    def test_condensation(self):
        self.cond(ngs)

    def test_nonlinear(self):
        self.nonlinear(ngs)

    def test_smallGeom(self):
        self.smallGeom(ngs)

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

    def test_twoval_grad(self):
        self.twoVars_grad(ngs)

    def test_direct_solvers(self):
        self.solver(ngs, "pardiso", None)
        self.solver(ngs, "pardiso", None, cond=True)
        self.solver(ngs, "pardisospd", None)
        self.solver(ngs, "sparsecholesky", None)
        self.solver(ngs, "masterinverse", None)
        self.solver(ngs, "umfpack", None)

    def test_iterative_solvers(self):
        self.solver(ngs, "cg", "gamg")
        self.solver(ngs, "cg", "gamg", cond=True)
        self.solver(ngs, "minres", "gamg")
        self.solver(ngs, "symmlq", "gamg")
        self.solver(ngs, "gmres", "gamg")
        self.solver(ngs, "bcgs", "gamg")

    def test_preconditioners(self):
        self.solver(ngs, "cg", "jacobi")
        self.solver(ngs, "cg", "bjacobi")
        self.solver(ngs, "cg", "ilu")
        self.solver(ngs, "cg", "icc")
        self.solver(ngs, "cg", "gamg")
        self.solver(ngs, "cg", "sor")

    def test_condensation(self):
        self.cond(ngs)

    def test_error(self):
        self.error(ngs)

    def test_random(self):
        self.random(ngs)