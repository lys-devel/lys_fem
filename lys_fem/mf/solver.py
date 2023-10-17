
from . import mfem


def generateSolver(fem, models):
    solverDict = {s.name: s for s in [StationarySolver]}
    return [solverDict[s.name](fem, s, models) for s in fem.solvers]


class StationarySolver:
    def __init__(self, fem, solver, models):
        self._fem = fem
        self._solver = solver
        self._models = models

    def execute(self):
        subSolvers = {"Linear Solver": LinearSolver}
        for i, sub in enumerate(self._solver.subSolvers):
            model = self._models[self._fem.models.index(sub.target)]
            solver = subSolvers[sub.name](model)
            x = solver.solve()
            mfem.print_("Step", i, ":", model.name, "model has been solved by", solver.name)

    @classmethod
    @property
    def name(cls):
        return "Stationary Solver"


class TimeDependentSolver:
    def __init__(self, solver):
        self._solver = solver

    @classmethod
    @property
    def name(cls):
        return "Stationary Solver"

    def execute(self):
        t, dt = 0.0, 0.01
        for ti in range(51):
            t, dt = model.step(t, dt)
            print(ti, t)
            # oper.SetParameters(u)


class LinearSolver:
    def __init__(self, model):
        self._model = model

    def solve(self):
        x, _ = self._model.getInitialValue()
        a = self._model.assemble_a()
        b = self._model.assemble_b()
        ess_tdof_list = self._model.essential_tdof_list()

        A = mfem.SparseMatrix()
        B = mfem.Vector()
        X = mfem.Vector()
        a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)

        solver, prec = self._getDefaultSolver(A)
        solver.Mult(B, X)  # Solve AX=B
        a.RecoverFEMSolution(X, b, x)
        return x

    def _getDefaultSolver(self, A, rel_tol=1e-8):
        solver, prec = mfem.getSolver()
        solver.iterative_mode = False
        solver.SetRelTol(rel_tol)
        solver.SetAbsTol(0.0)
        solver.SetMaxIter(100)
        solver.SetPrintLevel(0)
        solver.SetPreconditioner(prec)
        solver.SetOperator(A)
        return solver, prec

    @property
    def name(self):
        return "Linear Solver"


class GeneralizedAlphaSolver(mfem.SecondOrderTimeDependentOperator):
    def __init__(self, model, value=10):
        super().__init__(model._fespace.GetTrueVSize(), 0)
        self._model = model
        self._initialized = False
        self._value = value

    def _initialize(self):
        self._ode_solver = mfem.GeneralizedAlpha2Solver(self._value)
        self._ode_solver.Init(self)

        x_gf, xt_gf = self._model.getInitialValue()
        self._x = mfem.Vector()
        self._xt = mfem.Vector()  # dx/dt
        x_gf.GetTrueDofs(self._x)
        xt_gf.GetTrueDofs(self._xt)

        self._K_op = self._model.assemble_a()
        self._M_op = self._model.assemble_m()
        ess_tdof_list = self._model.essential_tdof_list()

        self._K0 = mfem.HypreParMatrix()
        self._K = mfem.HypreParMatrix()
        dummy = mfem.intArray()
        self._K_op.FormSystemMatrix(dummy, self._K0)
        self._K_op.FormSystemMatrix(ess_tdof_list, self._K)

        self._M = mfem.HypreParMatrix()
        self._M_op.FormSystemMatrix(ess_tdof_list, self._M)

        self._M_solver, self._M_prec = self._getDefaultSolver(self._M)
        self._T = None

        self._initialized = True

    def step(self, t, dt):
        if not self._initialized:
            self._initialize()
        return self._ode_solver.Step(self._x, self._xt, t, dt)

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
            self._T_solver, self._T_prec = self._getDefaultSolver(self._T)
        z = mfem.Vector(u.Size())
        self._K0.Mult(u, z)
        z.Neg()

        # iterate over Array<int> :D
        for j in self._model.essential_tdof_list():
            z[j] = 0.0

        self._T_solver.Mult(z, d2udt2)

    def _getDefaultSolver(self, A, rel_tol=1e-8):
        prec = mfem.HypreBoomerAMG(A)
        solver = mfem.CGSolver(MPI.COMM_WORLD)
        solver.iterative_mode = False
        solver.SetRelTol(rel_tol)
        solver.SetAbsTol(0.0)
        solver.SetMaxIter(100)
        solver.SetPrintLevel(0)
        solver.SetPreconditioner(prec)
        solver.SetOperator(A)
        return solver, prec
