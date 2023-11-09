from . import mfem
from .models import MFEMLinearModel


class _SubSolverBase:
    def __init__(self, sol, solver, prec=None):
        self._sol = sol
        self.solver, self.prec = mfem.getSolver(solver=solver, prec=prec)
        self.newton = NewtonSolver(self)

    def SetOperator(self, A):
        self.solver.SetOperator(A)

    def __mul__(self, b):
        res = mfem.Vector(b.Size())
        self.solver.Mult(b, res)
        return res

    def update(self, x):
        if not isinstance(self.model, MFEMLinearModel):
            self.model.update(x)

    def solve(self, model):
        self.model = model
        x0 = self.model.x0
        self.model.update(x0)
        if isinstance(model, MFEMLinearModel):
            self.newton.setMaxIteration(1)
        x = self.newton.solve(self, x0)
        return model.RecoverFEMSolution(x)

    def __call__(self, x):
        K, b = self.model.K, self.model.b
        return K * x - b

    def grad(self, x):
        K, b = self.model.DK, self.model.b
        return K

    @property
    def name(self):
        return self._sol.name


class CGSolver(_SubSolverBase):
    def __init__(self, sol):
        super().__init__(sol, solver="CG", prec="GS")


class GMRESSolver(_SubSolverBase):
    def __init__(self, sol):
        super().__init__(sol, solver="GMRES", prec="GS")


class NewtonSolver:
    def __init__(self, subSolver, max_iter=100):
        self._solver = subSolver
        self._max_iter = max_iter

    def setMaxIteration(self, max_iter):
        self._max_iter = max_iter

    def solve(self, F, x, eps=1e-8):
        Ji = self._solver
        for i in range(self._max_iter):
            J = F.grad(x)
            Ji.SetOperator(J)
            dx = Ji * F(x)
            x = x - dx
            if dx.Norml2() < eps:
                break
            F.update(x)
        return x


subSolvers = {"CG Solver": CGSolver, "GMRES Solver": GMRESSolver}
