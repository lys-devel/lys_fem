import os
import shutil
import numpy as np
import sympy as sp
from . import mfem, weakform, coef

from .solvers_fem import createFEMSolver
from .solvers_time import StationaryEquation, createTimeDependentEquation


def generateSolver(fem, mesh, models):
    solvers = {"Stationary Solver": StationarySolver, "Time Dependent Solver": TimeDependentSolver}
    return [solvers[s.name](fem, s, mesh, models, "Solver" + str(i)) for i, s in enumerate(fem.solvers)]


class SolverBase:
    def __init__(self, dirname):
        self._dirname = "Solutions/" + dirname
        if mfem.isRoot:
            if os.path.exists(self._dirname):
                shutil.rmtree(self._dirname)
        os.makedirs(self._dirname, exist_ok=True)

    def exportMesh(self, mesh):
        meshes = mfem.getMesh(mesh)
        for i, m in enumerate(meshes):
            mfem.saveData(self._dirname + "/mesh" + str(i) + ".npz", m.dictionary())

    def exportSolution(self, index, solution):
        mfem.saveData(self._dirname + "/data" + str(index), solution)


class StationarySolver(SolverBase):
    def __init__(self, fem, femSolver, mesh, models, dirname):
        super().__init__(dirname)
        self._femSolver = femSolver
        self._mesh = mesh
        self._fem = fem
        self._solver = createFEMSolver(femSolver.solver)
        self._model = CompositeModel(mesh, [models[fem.models.index(m)] for m in femSolver.models], StationaryTimeSolver)
        #self._equation = StationaryEquation(self._model)

    def execute(self):
        self.exportMesh(self._mesh)
        self.exportSolution(0, self._model.solution)

        sol = self._model.solve(self._solver)
        self.exportSolution(1, sol)
        mfem.print_("Stationary problem has been solved")

    @property
    def name(self):
        return self._femSolver.name

class TimeDependentSolver(SolverBase):
    def __init__(self, fem, tSolver, mesh, models, dirname):
        super().__init__(dirname)
        self._mesh = mesh
        self._tSolver = tSolver
        self._model = CompositeModel(mesh, [models[fem.models.index(m)] for m in tSolver.models], BackwardEulerSolver)
        self._solver = createFEMSolver(tSolver.solver.femSolver)
        #self._equation = createTimeDependentEquation(tSolver.solver, self._model)

    def execute(self):
        self.exportMesh(self._mesh)
        self.exportSolution(0, self._model.solution)
        t = 0
        for i, dt in enumerate(self._tSolver.getStepList()):
            mfem.print_("timestep", i, ", t =", t)
            #sol = self._equation.solve(dt)
            sol = self._model.solve(self._solver, dt)
            self.exportSolution(i + 1, sol)
            t = t + dt

    @property
    def name(self):
        return self._tSolver.name


class StationaryTimeSolver:
    @classmethod
    def discretizeTime(cls, wf, trials):
        t = weakform.t
        for trial in trials:
            wf = wf.subs({trial.diff(t).diff(t): 0, trial.diff(t): 0})
        return wf

class BackwardEulerSolver:
    @classmethod
    def discretizeTime(cls, wf, trials):
        t, dt, prev = weakform.t, weakform.dt, weakform.prev
        for trial in trials:
            wf = wf.subs({trial.diff(t).diff(t): 0, trial.diff(t): (trial-prev(trial))/dt})
        return wf


class CompositeModel:
    def __init__(self, mesh, models, tSolver):
        self._mesh = mesh
        self._models = models
        weakform_t = tSolver.discretizeTime(self.weakform, self.trialFunctions)
        self._nonlinear = self.__checkNonlinear(weakform_t, self.trialFunctions)
        if self._nonlinear:
            self._parser = weakform.WeakformParser(weakform_t, self.trialFunctions, self.coefficient)
        else:
            self._parser = weakform.LinearWeakformParser(weakform_t, self.trialFunctions, self.coefficient)
        self.x0 = self._parser.initialValue()

    def __checkNonlinear(self, wf, trials):
        vars = []
        for trial in trials:
            vars.append(trial)
            for gt in weakform.grad(trial):
                vars.append(gt)
        p = sp.poly(wf, *vars)
        for order in p.as_dict().keys():
            if sum(order) > 1:
                return True
        return False

    def update(self, x):
        self.K, self.b, self.gK = self._parser.update(x, self.dt, self._coeffs)

    def solve(self, solver, dt=1):
        self.dt = dt
        if not self.isNonlinear:
            solver.setMaxIteration(1)
        self._updateCoefficients()
        self.x0 = solver.solve(self, self.x0)
        return self.solution
    
    def _updateCoefficients(self):
        self._coeffs = {"dt": coef.generateCoefficient(self.dt, self._mesh.SpaceDimension())}
        x0 = mfem.BlockVector(self.x0, self._block_offset)
        for i, trial in enumerate(self.trialFunctions):
            p = weakform.prev(trial)
            gf = mfem.GridFunction(trial.mfem.space)
            gf.SetFromTrueDofs(x0.GetBlock(i))
            self._coeffs[str(p)] = coef.generateCoefficient(gf, self._mesh.SpaceDimension())
    
    def __call__(self, x):
        K, b = self.K, self.b
        res = mfem.Vector(x.Size())
        K.Mult(x, res)
        res -= b
        return res

    def grad(self, x):
        return self.gK

    @property
    def timeUnit(self):
        return 1

    @property
    def isNonlinear(self):
        return self._nonlinear

    def printVector(self, vec):
        for i, t in enumerate(self.trialFunctions):
            gf = mfem.GridFunction(t.mfem.space)
            self._mesh.GetNodes(gf)
            print(str(t).replace("trial_", ""), [v[0] for v in zip(vec.GetBlock(i).GetDataArray())])

    def dualToPrime(self, vec):
        self.vb = mfem.BlockVector(vec, self._block_offset)
        self._res_dtp = mfem.BlockVector(self._block_offset)
        self._primes = [trial.mfem.dualToPrime(self.vb.GetBlock(i)) for i, trial in enumerate(self.trialFunctions)]
        for i, p in enumerate(self._primes):
            self._res_dtp.GetBlock(i).Set(1, p)
        return self._res_dtp

    def getNodalValue(self, vec):
        vb = mfem.BlockVector(vec, self._block_offset)
        res = []
        for i, tri in enumerate(self.trialFunctions):
            gf = mfem.GridFunction(tri.mfem.space)
            gf.SetFromTrueDofs(vb.GetBlock(i))
            r = mfem.getData(gf, self._mesh)[:,0]
            res.append(r)
        return np.array(res)

    @property
    def _block_offset(self):
        offsets =mfem.intArray([0]+[t.mfem.space.GetTrueVSize() for t in self.trialFunctions])
        offsets.PartialSum()
        return offsets

    @property
    def weakform(self):
        return sum(m.weakform for m in self._models)

    @property
    def trialFunctions(self):
        result = []
        for m in self._models:
            result.extend(m.trialFunctions)
        return result

    @property
    def coefficient(self):
        result = {}
        for m in self._models:
            result.update(m.coefficient)
        return result
    
    @property
    def solution(self):
        result = {}
        x0 = mfem.BlockVector(self.x0, self._block_offset)
        for i, t in enumerate(self.trialFunctions):
            name = t.func.name.replace("trial_", "")
            gf = mfem.GridFunction(t.mfem.space)
            gf.SetFromTrueDofs(x0.GetBlock(i))
            result[name] = mfem.getData(gf, self._mesh)
        return result