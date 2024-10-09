import os
import shutil
import numpy as np

import ngsolve
from ngsolve import BilinearForm, LinearForm, CoefficientFunction, VectorH1, TaskManager, SetNumThreads

from . import mpi, time, util

SetNumThreads(16)

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
        self._coefs = [[self.__toFunc(self[n][i], "t"*i) for i in range(3)] for n in range(nlog)]
        self.update(model.initialValue(use_a))

    def __toFunc(self, x, pre=""):
        res = []
        n = 0
        for v in self._model.variables:
            if v.size == 1 and v.isScalar:
                res.append(util.NGSFunction(v.scale*x.components[n], name=v.name+pre+"0", tdep=True))
            else:
                res.append(util.NGSFunction(v.scale*CoefficientFunction(tuple(x.components[n:n+v.size])), name=v.name+pre+"0", tdep=True))
            n+=v.size           
        return res

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
        return self._coefs[n][0]

    def V(self, n=0):
        return self._coefs[n][1]

    def A(self, n=0):
        return self._coefs[n][2]


    @property
    def finiteElementSpace(self):
        return self._model.finiteElementSpace


class _Operator:
    def __init__(self, model, integ, sols, symbols, dti, solver="pardiso", prec=None):
        wf = self.__prepareWeakform(model, integ, sols, symbols, dti)
        self._model = model
        self._symbols = symbols
        self._solver = solver
        self._fes = model.finiteElementSpace
        self.__setCoupling()

        self._blf, self._lf = BilinearForm(self._fes), LinearForm(self._fes)
        lhs, rhs = wf.lhs, wf.rhs
        if lhs.valid:
            self._blf += lhs.eval()
        if rhs.valid:
            self._lf += rhs.eval()
        if prec is not None:
            self._prec = ngsolve.Preconditioner(self._blf, prec)
        self._nl_lhs = lhs.isNonlinear
        self._nl_rhs = rhs.isNonlinear
        self._tdep_lhs = lhs.isTimeDependent
        self._tdep_rhs = rhs.isTimeDependent
        self._init  = False

    def __prepareWeakform(self, model, integ, sols, symbols, dti):
        wf = model.weakforms()
        if symbols is not None:
            d = {}
            for v, X, V in zip(model.variables, sols.X(), sols.V()):
                if v.name not in symbols:
                    d[v.trial] = X
                    d[v.trial.t] = V
                    d[v.test] = 0
                    d[util.grad(v.test)] = 0
            wf = wf.replace(d)
        wf = integ.generateWeakforms(wf, model, sols, dti)
        return wf

    def __call__(self, x):
        with TaskManager():
            if self._nl_rhs:
                self._lf.Assemble()
            if self._nl_lhs:
                return self._blf.Apply(x) + self._lf.vec
            else:
                return self._blf.mat * x  + self._lf.vec

    def Jacobian(self, x):
        with TaskManager():
            if self._nl_lhs:
                self._blf.AssembleLinearization(x)
                res = self.__inverse(self._blf.mat)
                return res
            else:
                return self._inv
        
    def update(self, dti):
        self.__setCoupling()
        with TaskManager():
            if (self._tdep_rhs or not self._init) and not self._nl_rhs:
                self._lf.Assemble()
            if (self._tdep_lhs or not self._init) and not self._nl_lhs:
                self._blf.Assemble()
                self._inv = self.__inverse(self._blf.mat)
        self._init = True

    def __inverse(self, mat):
        with TaskManager():
            if self._solver in ["pardiso", "sparsecholesky", "masterinverse"]:
                return mat.Inverse(self._fes.FreeDofs(), self._solver)
            if self._solver == "CG":
                return ngsolve.CGSolver(mat, self._prec.mat)
            if self._solver == "GMRES":
                return ngsolve.GMRESSolver(mat, self._prec.mat)

    def __setCoupling(self):
        if self._symbols is None:
            return
        n = 0
        for v in self._model.variables:
            if v.name in self._symbols:
                t = ngsolve.COUPLING_TYPE.WIREBASKET_DOF
            else:
                t = ngsolve.COUPLING_TYPE.UNUSED_DOF
            for j in range(n, n+v.size):
                self._fes.SetCouplingType(self._fes.Range(j), t)
            n += v.size
        ngsolve.Compress(self._fes)
        
    @property
    def isNonlinear(self):
        return self._nl_lhs or self._nl_rhs
    


class SolverBase:
    def __init__(self, obj, mesh, model, dirname, variableStep=False):
        self._obj = obj
        self._mesh = mesh
        self._model = model
        self._mat = model.materials
        self._mat.const.dti.tdep = variableStep
        self.__prepareDirectory(dirname)

        self._integ = self.__prepareIntegrator(obj)
        self._sols = _Solution(model, self._integ.use_a)
        self._ops = [_Operator(model, self._integ, self._sols, step.variables, self._mat.const.dti, solver=step.solver, prec=step.preconditioner) for step in obj.steps]
        self._x = self._sols.copy()
        self._step = 0

    @np.errstate(divide='ignore', invalid="ignore")
    def solve(self, dti=0):
        self._mat.updateSolutionFields(self._step)
        self._step += 1
        x0 = self._x.vec.CreateVector(copy=True)
        self._mat.const.dti.set(dti)
        for op, step in zip(self._ops, self._obj.steps):
            if step.deformation is not None:
                deform = {v.name: x0 for v, x0 in zip(self._model.variables, self._sols.X())}[step.deformation]
                space = VectorH1(self._mesh)
                gf = util.GridFunction(space, deform.eval())
                self._mesh.SetDeformation(gf)
                print("deformation set")
            op.update(dti)
            self._x.vec.data = newton(op, self._x.vec.CreateVector(copy=True))
            self._sols.update(self._integ.updateSolutions(self._x, self._sols, self._mat.const.dti.get()))
        return self.__calcDifference(self._x.vec, x0)

    def __calcDifference(self, x, x0):
        x0.data = x0.data - x.data
        return np.divide(x0.InnerProduct(x0), x.InnerProduct(x))
            
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
        m.ngmesh.Save(self._dirname + "/mesh0.vol")

    def exportSolution(self, index):
        self._sols.save(self._dirname + "/ngs" + str(index))

    @property
    def solution(self):
        return self._sols

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
        super().__init__(obj, mesh, model, dirname, variableStep=True)
        self._tSolver = obj

    def execute(self):
        #self.exportMesh(self._mesh)
        self.exportSolution(0)

        t = 0
        dt, dx_ref = self._tSolver.dt0, self._tSolver.dx
        for i in range(1,100):
            util.t.set(t)
            dx = self.solve(1/dt)
            t = t + dt
            mpi.print_("Step", i, ", t = {:3e}".format(t), ", dt = {:3e}".format(dt), ", dx = {:3e}".format(dx))
            self.exportSolution(i)
            if dt == np.inf:
                break
            dt *= min(np.sqrt(dx_ref/dx), 3)
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
            util.t.set(t)
            dx = self.solve(1/dt)
            mpi.print_("Timestep", i, ", t = {:3e}".format(t), ", dx = {:3e}".format(dx))
            self.exportSolution(i + 1)
            t = t + dt


def newton(F, x, eps=1e-5, max_iter=30):
    if not F.isNonlinear:
        max_iter=1
    dx = x.CreateVector()
    for i in range(max_iter):
        with ngsolve.TaskManager():
            dx.data = F.Jacobian(x)*F(x)
        x -= dx
        R = np.sqrt(np.divide(dx.InnerProduct(dx), x.InnerProduct(x)))
        print("R =", R)
        if R < eps:
            if i!=0:
                mpi.print_("[Newton solver] Converged in", i, "steps.")
            return x
    if max_iter !=1:
        raise RuntimeError("[Newton solver] NOT Converged in " + str(i) + " steps.")
    return x
