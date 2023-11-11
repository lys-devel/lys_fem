from . import mfem


class FEMSolverBase:
    def __init__(self, solver, prec=None):
        self.solver, self.prec = mfem.getSolver(solver=solver, prec=prec)

    def SetOperator(self, A):
        self.solver.SetOperator(A)

    def __mul__(self, b):
        res = mfem.Vector(b.Size())
        self.solver.Mult(b, res)
        return res


class CGSolver(FEMSolverBase):
    def __init__(self, sol):
        super().__init__(solver="CG", prec="GS")


class GMRESSolver(FEMSolverBase):
    def __init__(self, sol):
        super().__init__(solver="GMRES", prec="GS")


class NewtonSolver:
    def __init__(self, subSolver, max_iter=100):
        self._solver = subSolver
        self._max_iter = max_iter

    def setMaxIteration(self, max_iter):
        self._max_iter = max_iter

    def solve(self, F, x, eps=1e-8):
        Ji = self._solver
        for i in range(self._max_iter):
            F.update(x)
            J = F.grad(x)
            Ji.SetOperator(J)
            dx = Ji * F(x)
            x = x - dx
            if dx.Norml2() < eps:
                break
        return x


def createFEMSolver(sol):
    subSolvers = {"CG Solver": CGSolver, "GMRES Solver": GMRESSolver}
    return NewtonSolver(subSolvers[sol.name](sol))
