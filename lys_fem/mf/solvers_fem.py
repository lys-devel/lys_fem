from . import mfem


class _SubSolverBase:
    def __init__(self, sol, solver, prec=None):
        self._sol = sol
        self.solver, self.prec = mfem.getSolver(solver=solver, prec=prec)

    def SetOperator(self, A):
        self.solver.SetOperator(A)

    def Mult(self, B, X):
        self.solver.Mult(B, X)

    def solve(self, model):
        A, B, X = model.assemble()
        self.solver.SetOperator(A)
        self.solver.Mult(B, X)
        return model.RecoverFEMSolution(X)

    @property
    def name(self):
        return self._sol.name


class CGSolver(_SubSolverBase):
    def __init__(self, sol):
        super().__init__(sol, solver="CG", prec="GS")


class NewtonSolver(_SubSolverBase):
    def __init__(self, sol):
        super().__init__(sol, solver="Newton")
        self.J_solver, self.J_prec = mfem.getSolver(solver="GMRES", prec="GS")
        self.solver.SetSolver(self.J_solver)


subSolvers = {"CG Solver": CGSolver, "Newton Solver": NewtonSolver}
