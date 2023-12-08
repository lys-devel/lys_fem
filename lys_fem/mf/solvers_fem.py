from . import mfem


class FEMSolverBase:
    def __init__(self, solver, prec=None):
        self.solver, self.prec = mfem.getSolver(solver=solver, prec=prec)

    def SetOperator(self, A):
        self.solver.SetOperator(A)

    def SetPreconditioner(self, prec):
        self.prec = prec
        self.solver.SetPreconditioner(prec)

    def __mul__(self, b):
        res = mfem.Vector(b.Size())
        self.solver.Mult(b, res)
        return res


class CGSolver(FEMSolverBase):
    def __init__(self, sol):
        super().__init__(solver="CG", prec=None)


class GMRESSolver(FEMSolverBase):
    def __init__(self, sol):
        super().__init__(solver="GMRES")


class NewtonSolver:
    def __init__(self, subSolver, max_iter=30):
        self._solver = subSolver
        self._max_iter = max_iter

    def setMaxIteration(self, max_iter):
        self._max_iter = max_iter

    def setPreconditioner(self, prec):
        self._solver.SetPreconditioner(prec)

    def solve(self, F, x, eps=1e-7):
        import time
        x = mfem.Vector(x)
        Ji = self._solver
        for i in range(self._max_iter):
            start = time.time()
            F.update(x)
            print("assemble", time.time()-start)
            J = F.grad(x)
            Ji.SetOperator(J)
            dx = Ji * F(x)
            x -= dx
            norm = mfem.getMax(x.Norml2())
            R = mfem.getMax(dx.Norml2())
            if norm != 0:
                R = R/norm
            if R < eps:
                print("Newton", i)
                return x
        if self._max_iter !=1:
            print("Newton solver does not converge.")
        return x


def createFEMSolver(sol):
    subSolvers = {"CG Solver": CGSolver, "GMRES Solver": GMRESSolver}
    return NewtonSolver(subSolvers[sol.name](sol))
