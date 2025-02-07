import os
import time
import shutil
import numpy as np

from . import mpi, util
from .operator import Operator, ConvergenceError

def generateSolver(fem, mesh, model):
    solvers = {"Stationary Solver": StationarySolver, "Relaxation Solver": RelaxationSolver, "Time Dependent Solver": TimeDependentSolver}
    result = []
    for i, s in enumerate(fem.solvers):
        sol = solvers[s.className]
        solver = sol(s, mesh, model, dirname="Solver" + str(i))
        result.append(solver)
    return result


class _Sol:
    def __init__(self, value):
        if isinstance(value, util.FiniteElementSpace):
            self._fes = value
            self._sols = (self._fes.gridFunction(), self._fes.gridFunction(), self._fes.gridFunction())
        else:
            self._fes = value[0].finiteElementSpace
            self._sols = value

    def __getitem__(self, index):
        return self._sols[index]
    
    def set(self, xva):
        if isinstance(xva, _Sol):
            xva = xva._sols
        for xi, yi in zip(self._sols, xva):
            if yi is not None:
                xi.vec.data = yi.vec
            else:
                xi.vec.data *= 0

    def copy(self):
        g = self._fes.gridFunction()
        for v in self._fes.variables:
            if v.type == "x":
                g.setComponent(v, util.SolutionFunction(v, self, 0).eval(self._fes))
            if v.type == "v":
                g.setComponent(v, util.SolutionFunction(v, self, 1).eval(self._fes))
            if v.type == "a":
                g.setComponent(v, util.SolutionFunction(v, self, 2).eval(self._fes))
        return g
    
    def project(self, fes):
        x, v, a = fes.gridFunction(), fes.gridFunction(), fes.gridFunction()
        for var in self._fes.variables:
            x.setComponent(var, util.SolutionFunction(var,self,0).eval(fes))
            v.setComponent(var, util.SolutionFunction(var,self,1).eval(fes))
            a.setComponent(var, util.SolutionFunction(var,self,2).eval(fes))
        return x,v,a

    def error(self, var):
        val = util.grad(util.SolutionFunction(var, self, 0))
        grids = []
        for d in range(3):
            g = self._fes.gridFunction()
            g.setComponent(var, val[d].eval(self._fes))
            g = g.toNGSFunctions(pre="_g")[var.name]
            grids.append(g)
        grids = util.NGSFunction(grids)
        err = np.sqrt(((grids-val)**2).integrate(self._fes, element_wise=True).NumPy())
        err = mpi.gatherArray(err)
        if mpi.isRoot:
            return np.concatenate(err)
        else:
            return [0]

    def save(self, path, mesh=False):
        self._sols[0].Save(path, parallel=mpi.isParallel())
        self._sols[1].Save(path+"_v", parallel=mpi.isParallel())
        self._sols[2].Save(path+"_a", parallel=mpi.isParallel())
        if mesh:
            self._fes.mesh.save(path+"_mesh.msh")

    @staticmethod
    def load(fes, path, parallel=None):
        if parallel is None:
            parallel = mpi.isParallel()
        x, v, a = (fes.gridFunction(), fes.gridFunction(), fes.gridFunction())
        if os.path.exists(path):
            x.Load(path, parallel)
        if os.path.exists(path+"_v"):
            v.Load(path+"_v", parallel)
        if os.path.exists(path+"_a"):
            a.Load(path+"_a", parallel)
        return _Sol((x,v,a))

    @property
    def finiteElementSpace(self):
        return self._fes

    @property
    def replaceDict(self):
        """
        Returns a dictionary that replace trial functions with corresponding solutions.
        """
        return {util.TrialFunction(v): util.SolutionFunction(v, self, 0) for v in self._fes.variables}


class _Solution:
    """
    Solution class stores the solutions and the time derivatives as grid function.
    The NGSFunctions based on the grid function is also provided by this class.
    """
    def __init__(self, fes, nlog=2, old=None):
        self._fes = fes
        self._sols = [_Sol(fes) for _ in range(nlog)]
        self._nlog = nlog

        if old is not None:
            for i, s in enumerate(self._sols):
                s.set(old[i].project(self._fes))

    def __getitem__(self, index):
        return self._sols[index]

    def initialize(self, model, op):
        x,v,a = model.initialValue(self._fes)
        self.__update(_Sol((x,v,a)))
        if op is not None:
            mpi.print_("\t======= Initial value calculation =======")
            mpi.print_(op)
            op.solve(self._sols[0][2])

    def reset(self):
        zero = self._fes.gridFunction()
        for _ in range(len(self._sols)):
            self.__update(_Sol((self._sols[0][0], zero, zero)))

    def updateSolution(self, model, x0):
        tdep = model.updater(self)
        fes = self._fes

        x, v, a = fes.gridFunction(), fes.gridFunction(), fes.gridFunction()
        x.vec.data = x0.vec

        fs = x0.toNGSFunctions("_new")
        d = {util.TrialFunction(v): fs[v.name] for v in self._fes.variables}

        for var in self._fes.variables:
            trial = util.TrialFunction(var)
            if trial in tdep:
                x.setComponent(var, tdep[trial].replace(d).eval(fes))
        for var in self._fes.variables:
            trial = util.TrialFunction(var)
            if trial.t in tdep:
                v.setComponent(var, tdep[trial.t].replace(d).eval(fes))
        for var in self._fes.variables:
            trial = util.TrialFunction(var)
            if trial.tt in tdep:
                a.setComponent(var, tdep[trial.tt].replace(d).eval(fes))

        self.__update(_Sol((x,v,a)))

    def __update(self, xva):
        for n in range(1, len(self._sols)):
            self._sols[-n].set(self._sols[-n+1])
        self._sols[0].set(xva)

    def X(self, var, n=0):
        return util.SolutionFunction(var, self._sols[n], 0)

    def V(self, var, n=0):
        return util.SolutionFunction(var, self._sols[n], 1)

    def A(self, var, n=0):
        return util.SolutionFunction(var, self._sols[n], 2)


class _DataStorage:
    def __init__(self, sols, dirname):
        self._sols = sols
        if dirname is not None:
            self._dirname = dirname
        else:
            self._dirname = None
        self._savemesh = False
        self.__init()
        self.save(0)

    def __init(self):
        if mpi.isRoot and self._dirname is not None:
            if os.path.exists(self._dirname):
                shutil.rmtree(self._dirname, ignore_errors=True)
            os.makedirs(self._dirname, exist_ok=True)

    def enableSaveMesh(self, b=True):
        self._savemesh=b

    def save(self, index):
        self._sols[0].save(self._dirname + "/ngs" + str(index), mesh=self._savemesh)


class SolverBase:
    def __init__(self, obj, mesh, model, dirname, timeDep=False, variableStep=False):
        self._obj = obj
        self._model = model
        self._mat = model.materials
        self._mat.const.dti.isTimeDependent = variableStep
        self._index = -1

        self._fes = util.FiniteElementSpace(model.variables, mesh, jacobi=self._mat.jacobi)
        self._sols = _Solution(self._fes)
        use_a = any([v.tt in model.weakforms() for v, _ in model.TnT.values()]) and timeDep
        if use_a:
            op = Operator(mesh, model, self._sols, obj.steps[0], type="initial")
        else:
            op = None
        self._sols.initialize(model, op)
        self._ops = [Operator(mesh, model, self._sols, step) for step in obj.steps]
        self._data = _DataStorage(self._sols, "Solutions/" + dirname)

    @np.errstate(divide='ignore', invalid="ignore")
    def solve(self, dti=0):
        start = time.time()
        if dti==0:
            self._sols.reset()
        self._index += 1
        self._mat.updateSolutionFields(self._index)
        self._mat.const.dti.set(dti)
        util.stepn.set(self._index)

        E0 = self.__calcDiff()
        self._step()
        E = self.__calcDiff()
        mpi.print_("\tTotal step time: {:.2f}".format(time.time()-start))
        mpi.print_()
        return np.divide(np.linalg.norm(E-E0), np.linalg.norm(E))

    def _step(self):
        x = self._sols[0].copy()
        for i, (op, step) in enumerate(zip(self._ops, self._obj.steps)):
            mpi.print_("\t=======Solver step", i+1, "=======")
            op.solve(x)
            mpi.print_()
        self.solutions.updateSolution(self._model, x)
        self._data.save(self._index+1)

    def __calcDiff(self):
        expr = self._obj.diff_expr
        if expr is None or expr=="":
            expr = self._model.variables[0].name
        d = self._sols[0].replaceDict
        return self._mat[expr].replace(d).integrate(self._fes)

    def updateMesh(self, mesh):
        self._fes = util.FiniteElementSpace(self._model.variables, mesh, jacobi=self._mat.jacobi)
        self._sols = _Solution(self._fes, old=self._sols)
        self._ops = [Operator(mesh, self._model, self._sols, step) for step in self._obj.steps]
        self._data.enableSaveMesh()

    def refineMesh(self, error):
        def compute_size_field(err, p=2, d=2):
            err = np.array(err)/np.median(err)
            return err**(-1/(1+p)) * p**(-1/(d*(1+p)))

        start = time.time()
        size = compute_size_field(error, d=self._fes.dimension)
        m = self._fes.mesh.refinedMesh(size, self._obj.adaptive_mesh)
        refine = time.time()-start
        self.updateMesh(m)
        total = time.time()-start
        mpi.print_("\t[AMR] Mesh updated: Num. of nodes =", self._fes.mesh.nodes, ", Time for refinement = {:.2f}".format(refine), ", Time for update = {:.2f}".format(total-refine))
        mpi.print_()

    @property
    def solutions(self):
        return self._sols

    @property
    def name(self):
        return self._obj.className

    @property
    def obj(self):
        return self._obj
    
    def __str__(self):
        res = ""
        for i, op in enumerate(self._ops):
            res += "\tStep " + str(1+i) + "\n"
            res += str(op)
        return res


class StationarySolver(SolverBase):
    def execute(self):
        self.solve()

        if self._obj.adaptive_mesh is not None:
            var = [v for v in self._model.variables if v.name==self._obj.adaptive_mesh.varName][0]
            error = self.solutions[0].error(var)
            mpi.print_("\t[AMR] Adaptive Mesh Refinement started. Initial error (max,min,mean) = {:.3e}, {:.3e}, {:.3e}".format(np.max(error), np.min(error), np.mean(error)))

            for n in range(self._obj.adaptive_mesh.maxiter):
                self.refineMesh(error)
                self.solve()
                error = self.solutions[0].error(var)

                mpi.print_("\t[AMR] Step "+str(n+2)+": Error (max,min,mean) = {:.3e}, {:.3e}, {:.3e}".format(np.max(error), np.min(error), np.mean(error)))


class RelaxationSolver(SolverBase):
    def __init__(self, obj, mesh, model, **kwargs):
        super().__init__(obj, mesh, model, timeDep=True, variableStep=True, **kwargs)
        self._tSolver = obj

    def execute(self):
        self._relax()
        if self._tSolver.adaptive_mesh is None:
            return
        var = [v for v in self._model.variables if v.name==self._obj.adaptive_mesh.varName][0]
        error = self.solutions[0].error(var)
        mpi.print_("\t[AMR] Adaptive Mesh Refinement started. Initial error (max,min,mean) = {:.3e}, {:.3e}, {:.3e}".format(np.max(error), np.min(error), np.mean(error)))
        mpi.print_()

        for n in range(self._obj.adaptive_mesh.maxiter):
            self.refineMesh(error)
            self._relax()
            error = self.solutions[0].error(var)

            mpi.print_("\t[AMR] Step "+str(n+2)+": Error (max,min,mean) = {:.3e}, {:.3e}, {:.3e}".format(np.max(error), np.min(error), np.mean(error)))
            mpi.print_()

    def _relax(self):
        t = 0
        dt, dx_ref = self._tSolver.dt0, self._tSolver.dx
        for i in range(1,self._tSolver.maxiter):
            util.t.set(t)
            dx, dt = self._solve(dt)
            t = t + dt
            mpi.print_("Step", i, ", t = {:3e}".format(t), ", dt = {:3e}".format(dt), ", dx = {:3e}".format(dx))
            if dx != 0:
                dt *= min(np.sqrt(dx_ref/abs(dx)), self._tSolver.maxStep)
            if dt < self._tSolver.dt0:
                dt = self._tSolver.dt0
            if dt > self._tSolver.dt_max:
                dt = self._tSolver.dt_max
            if dt == self._tSolver.dt_max and dx < self._tSolver.tolerance:
                mpi.print_("[Relaxation solver] Converged in", i, "steps")
                return

    def _solve(self, dt):
        try:
            return self.solve(1/dt), dt
        except ConvergenceError:
            mpi.print_("Convergence problem detected. Time step is changed to " + str(dt/2))
            self._index -= 1
            return self._solve(dt/2)


class TimeDependentSolver(SolverBase):
    def __init__(self, obj, mesh, model, **kwargs):
        super().__init__(obj, mesh, model, timeDep=True, **kwargs)
        self._tSolver = obj

    def execute(self):
        t = 0
        for i, dt in enumerate(self._tSolver.getStepList()):
            util.t.set(t)
            dx = self.solve(1/dt)
            t = t + dt
            mpi.print_("Timestep", i, ", t = {:3e}".format(t), ", dx = {:3e}".format(dx))
            if self._tSolver.adaptive_mesh is not None:
                var = [v for v in self._model.variables if v.name==self._obj.adaptive_mesh.varName][0]
                error = self.solutions[0].error(var)

                mpi.print_("\t[AMR] Error (max,min,mean) = {:.3e}, {:.3e}, {:.3e}".format(np.max(error), np.min(error), np.mean(error)))
                self.refineMesh(error)
                mpi.print_()


