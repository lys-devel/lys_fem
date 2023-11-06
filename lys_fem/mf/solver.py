import os
import shutil
import numpy as np
from . import mfem


def generateSolver(fem, mesh, models):
    solvers = {"Stationary Solver": StationarySolver, "Time Dependent Solver": TimeDependentSolver}
    return [solvers[s.name](fem, s, mesh, models, "Solver" + str(i)) for i, s in enumerate(fem.solvers)]


class SolverBase:
    def __init__(self, femSolver, mesh, models, dirname):
        self._subSolvers = {"Linear Solver": CGSolver, "CG Solver": CGSolver, "Newton Solver": NewtonSolver}
        self._tSolvers = {"Generalized Alpha Solver": GeneralizedAlphaSolver, "Backward Euler Solver": BackwardEulerSolver}
        self._femSolver = femSolver
        self._mesh = mesh
        self._models = models
        self._dirname = "Solutions/" + dirname
        if mfem.isRoot:
            if os.path.exists(self._dirname):
                shutil.rmtree(self._dirname)
        os.makedirs(self._dirname, exist_ok=True)

    def exportMesh(self, fec):
        meshes = mfem.getMesh(fec, self._mesh)
        for i, m in enumerate(meshes):
            mfem.saveData(self._dirname + "/mesh" + str(i) + ".npz", m.dictionary())

    def exportInitialValues(self):
        sol = {}
        for m in self._models:
            sol[m.variableName] = mfem.getData(m.getInitialValue()[0], self._mesh)
        mfem.saveData(self._dirname + "/data0.npz", sol)

    def exportSolution(self, index, solution):
        mfem.saveData(self._dirname + "/data" + str(index), solution)

    @property
    def name(self):
        return self._femSolver.name


class StationarySolver(SolverBase):
    def __init__(self, fem, femSolver, mesh, models, dirname):
        super().__init__(femSolver, mesh, models, dirname)
        self._fem = fem
        self._solver = [self._subSolvers[s.name](s) for s in femSolver.subSolvers]
        self._models = [models[fem.models.index(m)] for m in femSolver.models]

    def execute(self, fec):
        self.exportMesh(fec)
        self.exportInitialValues()
        sol = {}
        for solver, model in zip(self._solver, self._models):
            x = solver.solve(model)
            sol[model.variableName] = mfem.getData(x, self._mesh)
            mfem.print_(model.name, "model has been solved by", solver.name)
        self.exportSolution(1, sol)


class TimeDependentSolver(SolverBase):
    def __init__(self, fem, femSolver, mesh, models, dirname):
        super().__init__(femSolver, mesh, models, dirname)
        self._fem = fem
        self._femSolver = femSolver
        self._models = [models[fem.models.index(m)] for m in femSolver.models]
        self._tsolver = []
        for s, m in zip(femSolver.subSolvers, self._models):
            sub1, sub2 = self._subSolvers[s.femSolver.name](s), self._subSolvers[s.femSolver.name](s)
            tsol = self._tSolvers[s.name](m.space.GetTrueVSize(), sub1, sub2)
            self._tsolver.append(tsol)

    def execute(self, fec):
        self.exportMesh(fec)
        self.exportInitialValues()
        t = 0
        for i, dt in enumerate(self._femSolver.getStepList()):
            mfem.print_("t =", t)
            sol = {}
            for model, solver in zip(self._models, self._tsolver):
                x = solver.step(model, t, dt)
                sol[model.variableName] = mfem.getData(x, self._mesh)
            sol["time"] = t
            self.exportSolution(i + 1, sol)
            t = t + dt

    @classmethod
    @property
    def name(cls):
        return "Time Dependent Solver"


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


class TimeDependentOperator(mfem.PyTimeDependentOperator):
    def __init__(self, size, solM, solT, ode_solver):
        super().__init__(size, 0)
        self.Mi = solM
        self.Ti = solT
        self._ode_solver = ode_solver
        self._ode_solver.Init(self)

    def step(self, model, t, dt):
        self.K, self.b, x = model.assemble_t(dt, self.Mi, self.Ti)
        self._ode_solver.Step(x, t, dt)
        return model.RecoverFEMSolution(x)

    def Mult(self, u, du_dt):
        # Compute: du_dt = M^{-1}*-K(u)
        z = mfem.Vector(u.Size())
        self.K.Mult(u, z)
        z.Neg()   # z = -z
        self.Mi.Mult(z, du_dt)

    def ImplicitSolve(self, dt, u, du_dt):
        # Solve the equation: du_dt = M^{-1}*[-K(u + dt*du_dt)]
        z = mfem.Vector(u.Size())
        self.K.Mult(u, z)
        z.Neg()
        self.Ti.Mult(z, du_dt)


class BackwardEulerSolver(TimeDependentOperator):
    def __init__(self, size, solver_M, solver_T):
        super().__init__(size, solver_M, solver_T, mfem.BackwardEulerSolver())


class GeneralizedAlphaSolver(mfem.SecondOrderTimeDependentOperator):
    def __init__(self, model, value=10):
        super().__init__(model._fespace.GetTrueVSize(), 0)
        self._model = model
        self._initialized = False
        self._value = value

    def initialize(self):
        self._ode_solver = mfem.GeneralizedAlpha2Solver(self._value)
        self._ode_solver.Init(self)

        self._x_gf, self._xt_gf = self._model.getInitialValue()
        self._x = mfem.Vector()
        self._xt = mfem.Vector()  # dx/dt
        self._x_gf.GetTrueDofs(self._x)
        self._xt_gf.GetTrueDofs(self._xt)

        self._K_op = self._model.assemble_a()
        self._M_op = self._model.assemble_m()
        ess_tdof_list = self._model.essential_tdof_list()

        self._K0 = mfem.SparseMatrix()
        self._K = mfem.SparseMatrix()
        dummy = mfem.intArray()
        self._K_op.FormSystemMatrix(dummy, self._K0)
        self._K_op.FormSystemMatrix(ess_tdof_list, self._K)

        self._M = mfem.SparseMatrix()
        self._M_op.FormSystemMatrix(ess_tdof_list, self._M)

        self._M_solver, self._M_prec = _getDefaultSolver(self._M)
        self._T = None

        self._initialized = True

    def step(self, t, dt):
        if not self._initialized:
            self.initialize()
        self._ode_solver.Step(self._x, self._xt, t, dt)
        self._x_gf.SetFromTrueDofs(self._x)
        return self._x_gf

    def Mult(self, u, du_dt, d2udt2):
        # Compute d2udt2 = M^{-1}*-K(u)
        z = mfem.Vector(u.Size())
        self._K.Mult(u, z)
        z.Neg()  # z = -z
        self._M_solver.Mult(z, d2udt2)

    def ImplicitSolve(self, fac0, fac1, u, dudt, d2udt2):
        # Solve the equation: d2udt2 = M^{-1}*[-K(u + fac0*d2udt2)]
        if self._T is None:
            self._T = mfem.Add(1.0, self._M, fac0, self._K)
            self._T_solver, self._T_prec = _getDefaultSolver(self._T)
        z = mfem.Vector(u.Size())
        self._K0.Mult(u, z)
        z.Neg()

        # iterate over Array<int> :D
        for j in self._model.essential_tdof_list():
            z[j] = 0.0

        self._T_solver.Mult(z, d2udt2)


def _getDefaultSolver(A, rel_tol=1e-8):
    solver, prec = mfem.getSolver()
    solver.iterative_mode = False
    solver.SetRelTol(rel_tol)
    solver.SetAbsTol(0.0)
    solver.SetMaxIter(10)
    solver.SetPrintLevel(0)
    solver.SetPreconditioner(prec)
    solver.SetOperator(A)
    return solver, prec
