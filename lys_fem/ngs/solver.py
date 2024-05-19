import os
import shutil
import numpy as np

from ngsolve import Parameter, BilinearForm, LinearForm, CoefficientFunction

from . import mpi, mesh, time, util


def generateSolver(fem, mesh, model):
    solvers = {"Stationary Solver": StationarySolver, "Relaxation Solver": RelaxationSolver, "Time Dependent Solver": TimeDependentSolver}
    result = []
    for i, s in enumerate(fem.solvers):
        sol = solvers[s.className]
        result.append(sol(s, mesh, model, "Solver" + str(i)))
    return result


class _Solution:
    def __init__(self, model, use_a, nlog=2):
        self._model = model
        fes = model.finiteElementSpace
        self._sols = [(util.GridFunction(fes), util.GridFunction(fes), util.GridFunction(fes)) for n in range(nlog)]
        self.update(model.initialValue(use_a))

    def update(self, xva):
        # Store old data
        for j in range(3):
            for n in range(1, len(self._sols)):
                self._sols[-n][j].vec.data = self._sols[-n+1][j].vec
        # Set new one
        for xi, yi in zip(self._sols[0], xva):
            if yi is not None:
                xi.vec.data = yi.vec

    def __getitem__(self, n):
        return self._sols[n]

    def copy(self):
        g = util.GridFunction(self._model.finiteElementSpace)
        g.vec.data = self._sols[0][0].vec
        return g
    
    def save(self, path):
        self._sols[0][0].Save(path, parallel=mpi.isParallel())

    def X(self, n=0):
        return self.__toFunc(self[n][0])

    def V(self, n=0):
        return self.__toFunc(self[n][1], "t")

    def A(self, n=0):
        return self.__toFunc(self[n][2], "tt")

    def __toFunc(self, x, pre=""):
        res = []
        n = 0
        for v in self._model.variables:
            if v.size == 1 and v.isScalar:
                res.append(util.NGSFunction(x.components[n], v.name+pre+"0"))
            else:
                res.append(util.NGSFunction(CoefficientFunction(tuple(x.components[n:n+v.size])), v.name+pre+"0"))
            n+=v.size           
        return res

    @property
    def finiteElementSpace(self):
        return self._model.finiteElementSpace


class _Operator:
    def __init__(self, model, integ, sols, symbols):
        wf = self.__prepareWeakform(model, integ, sols, symbols)
        self._fes = model.finiteElementSpace
        self._blf, self._lf = BilinearForm(self._fes), LinearForm(self._fes)
        self._blf += wf.lhs.eval()
        self._lf += wf.rhs.eval()
        self._nl = model.isNonlinear
        self._init  = False

    def __prepareWeakform(self, model, integ, sols, symbols):
        self._dti = Parameter(-1)
        wf = model.weakforms()
        print(symbols, wf)
        d = {}
        for v in model.variables:
            if symbols is None:
                continue
            elif v.name not in symbols:
                d[v.trial] = v.trial.value
                d[v.trial.t] = util.NGSFunction()
                d[v.trial.tt] = util.NGSFunction()
                d[v.test] = util.NGSFunction()
                d[util.grad(v.trial)] = util.NGSFunction()
                d[util.grad(v.test)] = util.NGSFunction()
        print("------------------")
        wf = integ.generateWeakforms(wf, model, sols, util.NGSFunction(self._dti,"dti"))
        print("result", wf)
        return wf

    def __call__(self, x):
        self._lf.Assemble()
        if self._nl:
            return self._blf.Apply(x) + self._lf.vec
        else:
            return self._blf.mat * x  + self._lf.vec

    def Jacobian(self, x):
        if self._nl:
            self._blf.AssembleLinearization(x)
            return self._blf.mat.Inverse(self._fes.FreeDofs(), "pardiso")
        else:
            return self._inv
        
    def update(self, dti):
        if self._dti.Get() != dti:
            self._dti.Set(dti)
        else:
            return
        if self._nl:
            return
        if not self._init:
            self._blf.Assemble()
            self._inv = self._blf.mat.Inverse(self._fes.FreeDofs(), "pardiso")
            self._init = True
    
    @property
    def isNonlinear(self):
        return self._nl
    
    @property
    def dti(self):
        return self._dti.Get()


class SolverBase:
    def __init__(self, obj, mesh, model, dirname):
        self._obj = obj
        self._mesh = mesh
        self.__prepareDirectory(dirname)

        self._solver = _NewtonSolver()
        self._integ = self.__prepareIntegrator(obj)
        self._sols = _Solution(model, self._integ.use_a)
        self._ops = [_Operator(model, self._integ, self._sols, step.variables) for step in obj.steps]
        self._x = self._sols.copy()

    @np.errstate(divide='ignore', invalid="ignore")
    def solve(self, dti=0):
        for op in self._ops:
            op.update(dti)
            self._x.vec.data = self._solver.solve(op, self._x.vec.CreateVector(copy=True))
            self._sols.update(self._integ.updateSolutions(self._x, self._sols, op.dti))

    def __prepareIntegrator(self, obj):
        if obj.method == "BackwardEuler":
            return time.BackwardEuler()
        elif obj.method == "NewmarkBeta":
            return time.NewmarkBeta()
        else:
            return time.GeneralizedAlpha(obj.method)

    def __prepareDirectory(self, dirname):
        self._dirname = "Solutions/" + dirname
        if mpi.isRoot:
            if os.path.exists(self._dirname):
                shutil.rmtree(self._dirname)
        os.makedirs(self._dirname, exist_ok=True)

    def exportMesh(self, m):
        mesh.exportMesh(m, self._dirname + "/mesh.npz")

    def exportSolution(self, index):
        self._sols.save(self._dirname + "/ngs" + str(index))

    @property
    def name(self):
        return self._obj.className


class StationarySolver(SolverBase):
    def execute(self):
        #self.exportMesh(self._mesh)
        self.exportSolution(0)

        self.solve()
        self.exportSolution(1)


class RelaxationSolver(SolverBase):
    def __init__(self, obj, mesh, model, dirname):
        super().__init__(obj, mesh, model, dirname)
        self._tSolver = obj

    def execute(self):
        #self.exportMesh(self._mesh)
        self.exportSolution(0)

        t = 0
        dt, dx = self._tSolver.dt0, self._tSolver.dx
        for i in range(1,1000):
            self.solve(1/dt)
            t = t + dt
            mpi.print_("Step", i, ", t = {:3e}".format(t), ", dt = {:3e}".format(dt), ", dx = {:3e}".format(self._solver.dx))
            self.exportSolution(i)
            if dt == np.inf:
                break
            dt *= np.sqrt(dx/self._solver.dx)
            if dt > self._tSolver.dt0*1e5:
                dt = np.inf


class TimeDependentSolver(SolverBase):
    def __init__(self, obj, mesh, model, dirname):
        super().__init__(obj, mesh, model, dirname)
        self._tSolver = obj

    def execute(self):
        #self.exportMesh(self._mesh)
        self.exportSolution(0)

        t = 0
        for i, dt in enumerate(self._tSolver.getStepList()):
            mpi.print_("Timestep", i, ", t = {:3e}".format(t), ", dt = {:3e}".format(dt), ", dx = {:3e}".format(self._solver.dx))
            self.solve(1/dt)
            self.exportSolution(i + 1)
            t = t + dt


class _NewtonSolver:
    def __init__(self, max_iter=30):
        self._max_iter = max_iter
        self._dx = 0

    def solve(self, F, x, eps=1e-5):
        if not F.isNonlinear:
            self._max_iter=1
        dx = x.CreateVector()
        x0 = x.CreateVector(copy=True)
        for i in range(self._max_iter):
            Fx = F(x)
            dx.data = F.Jacobian(x)*Fx
            x.data -= dx
            R = np.sqrt(np.divide(dx.InnerProduct(dx), x.InnerProduct(x)))
            if R < eps:
                if i!=0:
                    mpi.print_("[Newton solver] Converged in", i, "steps.")
                self._setDifference(x, x0)
                return x
        if self._max_iter !=1:
            mpi.print_("[Newton solver] NOT Converged in", i, "steps.")
        self._setDifference(x, x0)
        return x

    def _setDifference(self, x, x0):
        x0.data = x0.data - x.data
        self._dx = np.divide(x0.InnerProduct(x0), x.InnerProduct(x))
 
    @property
    def dx(self):
        return self._dx