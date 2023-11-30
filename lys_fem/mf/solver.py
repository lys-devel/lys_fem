import os
import shutil
from . import mfem, weakform

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
        self._model = CompositeModel(mesh, [models[fem.models.index(m)] for m in femSolver.models])
        self._equation = StationaryEquation(self._model)

    def execute(self):
        self.exportMesh(self._mesh)
        self.exportSolution(0, self._model.solution)

        sol = self._equation.solve(self._solver)
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
        self._model = CompositeModel(mesh, [models[fem.models.index(m)] for m in tSolver.models])
        self._equation = createTimeDependentEquation(tSolver.solver, self._model)

    def execute(self):
        self.exportMesh(self._mesh)
        self.exportSolution(0, self._model.solution)
        t = 0
        for i, dt in enumerate(self._tSolver.getStepList()):
            mfem.print_("timestep", i, ", t =", t)
            sol = self._equation.solve(dt)
            self.exportSolution(i + 1, sol)
            t = t + dt

    @property
    def name(self):
        return self._tSolver.name
    

class CompositeModel:
    def __init__(self, mesh, models):
        self._mesh = mesh
        self._models = models
        self.assemble()

    def update(self, x):
        pass

    @property
    def timeUnit(self):
        return 1

    def assemble(self):
        self._parser = weakform.WeakformParser(self.weakform, self.trialFunctions, self.coefficient)
        self.M, self.K, self.x0, self.b = self._parser.update()
        self.grad_Mx = self.M
        self.grad_Kx = self.K
        self.grad_b = None

    def printVector(self, vec):
        for i, t in enumerate(self.trialFunctions):
            gf = mfem.GridFunction(t.mfem.space)
            self._mesh.GetNodes(gf)
            print(str(t).replace("trial_", ""), [v[0] for v in zip(vec.GetBlock(i).GetDataArray())])

    def dualToPrime(self, vec):
        offsets =mfem.intArray([0]+[t.mfem.space.GetTrueVSize() for t in self.trialFunctions])
        offsets.PartialSum()
        vb = mfem.BlockVector(vec, offsets)
        res = mfem.BlockVector(offsets)
        for i, trial in enumerate(self.trialFunctions):
            res.GetBlock(i).Set(1, trial.mfem.dualToPrime(vb.GetBlock(i)))
        return res

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
        pos = 0
        for t in self.trialFunctions:
            name = t.func.name.replace("trial_", "")
            size = t.mfem.space.GetTrueVSize()

            gf = mfem.GridFunction(t.mfem.space)
            gf.SetFromTrueDofs(self.x0[pos:size])
            result[name] = mfem.getData(gf, self._mesh)
            pos = pos + size
        return result