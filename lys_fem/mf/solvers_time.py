from . import mfem
from .models import MFEMLinearModel


class StationaryEquation:
    def __init__(self, model):
        self.model = model
        self.model.update(self.model.x0)

    def update(self, x):
        if not isinstance(self.model, MFEMLinearModel):
            self.model.update(x)

    def solve(self, solver):
        if isinstance(self.model, MFEMLinearModel):
            solver.setMaxIteration(1)
        x = solver.solve(self, self.model.x0)
        return self.model.RecoverFEMSolution(x)

    def __call__(self, x):
        K, b = self.model.K, self.model.b
        return K * x - b

    def grad(self, x):
        K, b = self.model.DK, self.model.b
        return K


class _TimeDependentSubSolverBase:
    def __init__(self, model):
        self.model = model

    def update(self, x):
        if not isinstance(self.model, MFEMLinearModel):
            self.model.update(x)

    def solve(self, solver, dt):
        if isinstance(self.model, MFEMLinearModel):
            solver.setMaxIteration(1)
        self.dt = dt
        x = solver.solve(self, self.model.x0)
        return self.model.RecoverFEMSolution(x)


class BackwardEulerSolver(_TimeDependentSubSolverBase):
    def __init__(self, sol, model):
        super().__init__(model)

    def __call__(self, x):
        M, K, b, x0, dt = self.model.M, self.model.K, self.model.b, self.model.x0, self.dt
        return M * (x - x0) + K * x * dt - b * dt

    def grad(self, x):
        M, K, b, dt = self.model.M, self.model.K, self.model.b, self.dt
        return M + K * dt


class _SecondOrderTimeDependentSubSolverBase(mfem.SecondOrderTimeDependentOperator):
    def __init__(self, size, solM, solT, ode_solver):
        super().__init__(size, 0)
        self.Mi = solM
        self.Ti = solT
        self._ode_solver = ode_solver
        self._ode_solver.Init(self)

    def step(self, model, t, dt):
        self._model = model
        self.K, self.b, x, xt = model.K, model.b, model.x0, model.xt0
        self._ode_solver.Step(x, xt, t, dt)
        return model.RecoverFEMSolution(x)

    def Mult(self, u, du_dt, d2udt2):
        # Compute d2udt2 = M^{-1}*-K(u)
        self.Mi = self._model.set_Mi(self.Mi)
        z = mfem.Vector(u.Size())
        self.K.Mult(u, z)
        z.Neg()   # z = -z
        self.Mi.Mult(z, du_dt)

    def ImplicitSolve(self, fac0, fac1, u, dudt, d2udt2):
        # Solve the equation: d2udt2 = M^{-1}*[-K(u + fac0*d2udt2)]
        self.Ti = self._model.set_Ti(self.Ti, fac0)
        z = mfem.Vector(u.Size())
        self.K.Mult(u, z)
        z.Neg()

        # iterate over Array<int> :D
        for j in self._model.essential_tdof_list():
            z[j] = 0.0

        self.Ti.Mult(z, d2udt2)


class GeneralizedAlphaSolver(_SecondOrderTimeDependentSubSolverBase):
    def __init__(self, sol, size, solver_M, solver_T):
        super().__init__(size, solver_M, solver_T, mfem.GeneralizedAlpha2Solver(1))


def createTimeDependentEquation(sol, model):
    tSolvers = {"Generalized Alpha Solver": GeneralizedAlphaSolver, "Backward Euler Solver": BackwardEulerSolver}
    return tSolvers[sol.name](sol, model)
