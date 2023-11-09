from . import mfem


class _TimeDependentSubSolverBase(mfem.PyTimeDependentOperator):
    def __init__(self, size, solM, solT, ode_solver):
        super().__init__(size, 0)
        self.Mi = solM
        self.Ti = solT
        self._ode_solver = ode_solver
        self._ode_solver.Init(self)

    def step(self, model, t, dt):
        self._model = model
        self.K, self.b, x = model.K, model.b, model.x0
        self._ode_solver.Step(x, t, dt)
        return model.RecoverFEMSolution(x)

    def Mult(self, u, du_dt):
        # Compute: du_dt = M^{-1}*-K(u)
        self.Mi = self._model.set_Mi(self.Mi)
        z = mfem.Vector(u.Size())
        self.K.Mult(u, z)
        z.Neg()   # z = -z
        self.Mi.Mult(z, du_dt)

    def ImplicitSolve(self, dt, u, du_dt):
        # Solve the equation: du_dt = M^{-1}*[-K(u + dt*du_dt)]
        self.Ti = self._model.set_Ti(self.Ti, dt)
        z = self.K*u
        z.Neg()
        self.Ti.Mult(z, du_dt)


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


class BackwardEulerSolver(_TimeDependentSubSolverBase):
    def __init__(self, sol, size, solver_M, solver_T):
        super().__init__(size, solver_M, solver_T, mfem.BackwardEulerSolver())

class BackwardEulerSolver2(mfem.PyOperator):
    def __init__(self, sol, size, solM, solT):
        self.Mi = solM
        self.Ti = solT

    def step(self, model, t, dt):
        self._model = model
        M, K, b, x = model.M, model.K, model.b, model.x0
        Ti = self._model.set_Ti(self.Ti, dt)

        z = mfem.Vector(x.Size())
        M.Mult(x, z)
        Ti.Mult(z, x)
        return model.RecoverFEMSolution(x)
    
    def Mult(self, x, y):
        model = self._model
        M, K, b = model.M, model.K, model.b
        Mu = mfem.Vector(x)
        Mu = Mu - x0
        tKu = K.Mult(x, tKu)
        

    def GetGradient(self, x):
        self.dA = mfem.SparseMatrix()
        self.da.FormSystemMatrix(self._model.essential_tdof_list(), self.dA)
        return self.dA


class BackwardEulerSolver3:
    def __init__(self, sol, size, solM, solT):
        self.Mi = solM
        self.Ti = solT

    def step(self, model, t, dt):
        self._model = model
        self._dt = dt
        self.x0 = model.x0
        x = self.solver.solve(x)
        return model.RecoverFEMSolution(x)
    
    def Mult(self, x):
        model = self._model
        M, K, b = model.M, model.K, model.b
        Mx = M*(x - self.x0)
        tKu = K*x*self.dt
        bt = b*self.dt
        return Mx+tKu-bt

    def GetGradient(self, x):
        self.dA = mfem.SparseMatrix()
        self.da.FormSystemMatrix(self._model.essential_tdof_list(), self.dA)
        return self.dA


class GeneralizedAlphaSolver(_SecondOrderTimeDependentSubSolverBase):
    def __init__(self, sol, size, solver_M, solver_T):
        super().__init__(size, solver_M, solver_T, mfem.GeneralizedAlpha2Solver(1))


tSolvers = {"Generalized Alpha Solver": GeneralizedAlphaSolver, "Backward Euler Solver": BackwardEulerSolver}
