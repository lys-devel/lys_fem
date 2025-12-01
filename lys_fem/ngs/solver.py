import os
import time
import shutil
import numpy as np

from . import mpi, util
from .solution import Solution

def generateSolver(fem, mesh, model, mat):
    solvers = {"Stationary Solver": StationarySolver, "Relaxation Solver": RelaxationSolver, "Time Dependent Solver": TimeDependentSolver}
    result = []
    for i, s in enumerate(fem.solvers):
        sol = solvers[s.className]
        solver = sol(s, mesh, model, mat, dirname="Solver" + str(i))
        result.append(solver)
    return result


class SolverBase:
    def __init__(self, obj, mesh, model, mat, dirname, timeDep=False, variableStep=False):
        self._obj = obj
        self._model = model
        self._mat = mat
        util.dti.isTimeDependent = variableStep
        self._index = -1

        self._fes = util.FiniteElementSpace(model.variables, mesh)
        self._sols = self._initializeSolution(obj.steps[0], model, timeDep)
        self._ops = [self._initializeSolver(self._fes, model, self._sols, step) for step in obj.steps]
        self._data = _DataStorage(self._sols, "Solutions/" + dirname)

    def _initializeSolution(self, step, model, timeDep):
        x, v, a = model.initialValue(self._fes, self._mat)
        wf = model.weakforms(self._mat)
        if any([util.trial(var).tt in wf for var in model.variables]) and timeDep:
            d = {}
            for var in model.variables:
                trial, test = util.trial(var), util.test(var)
                X, V = util.GridField(x, var), util.GridField(v, var)
                if trial.tt in wf:
                    d.update({trial: X, trial.t: V, util.grad(trial): util.grad(X), trial.tt: trial})
                else:
                    d.update({trial: X, trial.t: V, util.grad(trial): util.grad(X), test: 0, util.grad(test): 0})
            op = util.Solver(self._fes.compress(step.variables), wf.replace(d), step.linear, step.nonlinear, parallel=mpi.isParallel())

            mpi.print_("\t======= Initial value calculation =======")
            mpi.print_(op)
            mpi.print_(op.solve(a))
        return Solution(self._fes, (x,v,a))
    
    def _initializeSolver(self, fes, model, sols, step):
        wf = model.weakforms(self._mat)

        # discretize time derivative of the weakform
        d = dict(model.discretize())
        if step.variables is not None:
            for v in fes.variables:
                trial, test = util.trial(v), util.test(v)
                if v.name not in step.variables:
                    d.update({trial: util.prev(trial), trial.t: util.prev(trial.t), trial.tt: util.prev(trial.tt), test:0, util.grad(test): 0})
        wf = wf.replace(d)

        # Replace previous values by the present solution
        wf = wf.replace(sols.replaceDict(trial=False, prev=True))

        return util.Solver(fes.compress(step.variables), wf, step.linear, step.nonlinear, parallel=mpi.isParallel())

    @np.errstate(divide='ignore', invalid="ignore")
    def solve(self, dti=0):
        start = time.time()
        if dti==0:
            self._sols.reset()
        self._index += 1
        self._obj.fem.randomFields.update()
        self._obj.fem.solutionFields.update(self._index)
        util.dti.set(dti)
        util.stepn.set(self._index)

        E0 = self.__calcDiff()
        self._step()
        E = self.__calcDiff()
        mpi.print_("\tTotal step time: {:.2f}".format(time.time()-start))
        mpi.print_()
        return np.divide(np.linalg.norm(E-E0), np.linalg.norm(E))

    def _step(self):
        x = self._sols.copyVector()
        for i, op  in enumerate(self._ops):
            mpi.print_("\t=======Solver step", i+1, "=======")
            mpi.print_(op.solve(x))
        self.solutions.updateSolution(self._model, x)
        self._data.save(self._index+1)

    def __calcDiff(self):
        expr = self._obj.diff_expr
        if expr is None or expr=="":
            expr = self._model.variables[0].name
        d = self._sols.replaceDict()
        return self._mat[expr].replace(d).integrate(self._fes)

    def updateMesh(self, mesh):
        self._fes = util.FiniteElementSpace(self._model.variables, mesh)
        self._sols = Solution(self._fes, old=self._sols)
        self._ops = [self._initializeSolver(self._fes, self._model, self._sols, step) for step in self._obj.steps]
        self._data.update(self._sols, True)

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

    def update(self, sols, updateMesh=True):
        self._sols = sols
        self._savemesh=updateMesh

    def save(self, index):
        self._sols.save(self._dirname + "/ngs" + str(index), mesh=self._savemesh)


class StationarySolver(SolverBase):
    def execute(self):
        self.solve()

        if self._obj.adaptive_mesh is not None:
            var = [v for v in self._model.variables if v.name==self._obj.adaptive_mesh.varName][0]
            error = self.solutions.error(var)
            mpi.print_("\t[AMR] Adaptive Mesh Refinement started. Initial error (max,min,mean) = {:.3e}, {:.3e}, {:.3e}".format(np.max(error), np.min(error), np.mean(error)))

            for n in range(self._obj.adaptive_mesh.maxiter):
                self.refineMesh(error)
                self.solve()
                error = self.solutions.error(var)

                mpi.print_("\t[AMR] Step "+str(n+2)+": Error (max,min,mean) = {:.3e}, {:.3e}, {:.3e}".format(np.max(error), np.min(error), np.mean(error)))


class RelaxationSolver(SolverBase):
    def __init__(self, obj, mesh, model, mat, **kwargs):
        super().__init__(obj, mesh, model, mat, timeDep=True, variableStep=True, **kwargs)
        self._tSolver = obj

    def execute(self):
        self._relax()
        if self._tSolver.adaptive_mesh is None:
            return
        var = [v for v in self._model.variables if v.name==self._obj.adaptive_mesh.varName][0]
        error = self.solutions.error(var)
        mpi.print_("\t[AMR] Adaptive Mesh Refinement started. Initial error (max,min,mean) = {:.3e}, {:.3e}, {:.3e}".format(np.max(error), np.min(error), np.mean(error)))
        mpi.print_()

        for n in range(self._obj.adaptive_mesh.maxiter):
            self.refineMesh(error)
            self._relax()
            error = self.solutions.error(var)

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
        except util.ConvergenceError:
            mpi.print_("Convergence problem detected. Time step is changed to " + str(dt/2))
            self._index -= 1
            return self._solve(dt/2)


class TimeDependentSolver(SolverBase):
    def __init__(self, obj, mesh, model, mat, **kwargs):
        super().__init__(obj, mesh, model, mat, timeDep=True, **kwargs)
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
                error = self.solutions.error(var)

                mpi.print_("\t[AMR] Error (max,min,mean) = {:.3e}, {:.3e}, {:.3e}".format(np.max(error), np.min(error), np.mean(error)))
                self.refineMesh(error)
                mpi.print_()


