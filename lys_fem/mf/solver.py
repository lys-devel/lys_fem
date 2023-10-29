import os
import shutil
from . import mfem


def generateSolver(fem, mesh, models):
    solverDict = {s.name: s for s in [StationarySolver, TimeDependentSolver]}
    return [solverDict[s.name](fem, mesh, s, models, "Solver" + str(i)) for i, s in enumerate(fem.solvers)]


class SolverBase:
    def __init__(self, mesh, models, dirname):
        self._subSolvers = {"Linear Solver": LinearSolver, "Generalized Alpha Solver": GeneralizedAlphaSolver, "Backward Euler Solver": BackwardEulerSolver}
        self._mesh = mesh
        self._models = models
        self._dirname = "Solutions/" + dirname
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


class StationarySolver(SolverBase):
    def __init__(self, fem, mesh, solver, models, dirname):
        super().__init__(mesh, models, dirname)
        self._fem = fem
        self._solver = solver
        self._models = models

    def execute(self, fec):
        self.exportMesh(fec)
        self.exportInitialValues()
        sol = {}
        for i, sub in enumerate(self._solver.subSolvers):
            model = self._models[self._fem.models.index(sub.target)]
            solver = self._subSolvers[sub.name](model)
            s = solver.solve()
            sol[sub.target.variableName] = mfem.getData(s, self._mesh)
            mfem.print_("Step", i, ":", model.name, "model has been solved by", solver.name)
        mfem.saveData(self._dirname + "/data1.npz", sol)

    @classmethod
    @property
    def name(cls):
        return "Stationary Solver"


class TimeDependentSolver(SolverBase):
    def __init__(self, fem, mesh, solver, models, dirname):
        super().__init__(mesh, models, dirname)
        self._fem = fem
        self._solver = solver

    def execute(self, fec):
        solvers = [self._subSolvers[sub.name](self._models[self._fem.models.index(sub.target)]) for sub in self._solver.subSolvers]
        self.exportMesh(fec)
        self.exportInitialValues()
        for s in solvers:
            s.initialize()
        t = 0
        for i, dt in enumerate(self._solver.getStepList()):
            mfem.print_("t =", t)
            sol = {}
            for s, sub in zip(solvers, self._solver.subSolvers):
                x = s.step(t, dt)
                sol[sub.variableName] = mfem.getData(x, self._mesh)
            sol["time"] = t
            mfem.saveData(self._dirname + "/data" + str(i + 1) + ".npz", sol)
            t = t + dt

    @classmethod
    @property
    def name(cls):
        return "Time Dependent Solver"


class LinearSolver:
    def __init__(self, model):
        self._model = model
        self.x, _ = self._model.getInitialValue()
        self.a = self._model.assemble_a()
        self.b = self._model.assemble_b()
        self.ess_tdof_list = self._model.essential_tdof_list()

    def solve(self):
        A = mfem.SparseMatrix()
        B = mfem.Vector()
        X = mfem.Vector()
        self.a.FormLinearSystem(self.ess_tdof_list, self.x, self.b, A, X, B)

        solver, prec = _getDefaultSolver(A)
        solver.Mult(B, X)  # Solve AX=B
        self.a.RecoverFEMSolution(X, self.b, self.x)
        return self.x

    @property
    def name(self):
        return "Linear Solver"


class BackwardEulerSolver(mfem.PyTimeDependentOperator):
    def __init__(self, model):
        super().__init__(model._fespace.GetTrueVSize(), 0)
        self._model = model
        self._initialized = False

    def initialize(self):
        self._ode_solver = mfem.BackwardEulerSolver()
        self._ode_solver.Init(self)

        self._x_gf, _ = self._model.getInitialValue()
        self._x = mfem.Vector()
        self._x_gf.GetTrueDofs(self._x)

        self._K_op = self._model.assemble_a()
        self._M_op = self._model.assemble_m()
        ess_tdof_list = self._model.essential_tdof_list()

        self._K = mfem.SparseMatrix()
        self._K_op.FormSystemMatrix(ess_tdof_list, self._K)

        self._M = mfem.SparseMatrix()
        self._M_op.FormSystemMatrix(ess_tdof_list, self._M)

        self._M_solver, self._M_prec = _getDefaultSolver(self._M)
        self._T = None

        self._initialized = True

    def step(self, t, dt):
        if not self._initialized:
            self.initialize()
        self._ode_solver.Step(self._x, t, dt)
        self._x_gf.SetFromTrueDofs(self._x)
        return self._x_gf

    def Mult(self, u, du_dt):
        # Compute: du_dt = M^{-1}*-K(u)
        z = mfem.Vector(u.Size())
        self._K.Mult(u, z)
        z.Neg()   # z = -z
        self.M_solver.Mult(z, du_dt)

    def ImplicitSolve(self, dt, u, du_dt):
        # Solve the equation: du_dt = M^{-1}*[-K(u + dt*du_dt)]
        if self._T is None:
            self._T = mfem.Add(1.0, self._M, dt, self._K)
            self._T_solver, self._T_prec = _getDefaultSolver(self._T)
        z = mfem.Vector(u.Size())
        self._K.Mult(u, z)
        z.Neg()
        self._T_solver.Mult(z, du_dt)


class GeneralizedAlphaSolver(mfem.SecondOrderTimeDependentOperator):
    def __init__(self, model, value=10):
        super().__init__(model._fespace.GetTrueVSize(), 0)
        self._model = model
        self._initialized = False
        self._value = value

    def initialize(self):
        import time
        start = time.time()
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
    solver.SetMaxIter(100)
    solver.SetPrintLevel(0)
    solver.SetPreconditioner(prec)
    solver.SetOperator(A)
    return solver, prec
