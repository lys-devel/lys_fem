import os
import shutil
import numpy as np
from . import mfem

from .solvers_fem import subSolvers
from .solvers_time import tSolvers


def generateSolver(fem, mesh, models):
    solvers = {"Stationary Solver": StationarySolver, "Time Dependent Solver": TimeDependentSolver}
    return [solvers[s.name](fem, s, mesh, models, "Solver" + str(i)) for i, s in enumerate(fem.solvers)]


class SolverBase:
    def __init__(self, femSolver, mesh, models, dirname):
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
        self._solver = [subSolvers[s.name](s) for s in femSolver.subSolvers]
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
            tsol = tSolvers[s.name](s)
            self._tsolver.append(tsol)

    def execute(self, fec):
        self.exportMesh(fec)
        self.exportInitialValues()
        t = 0
        for i, dt in enumerate(self._femSolver.getStepList()):
            mfem.print_("t =", t)
            sol = {}
            for model, solver in zip(self._models, self._tsolver):
                x = solver.step(model, dt)
                sol[model.variableName] = mfem.getData(x, self._mesh)
            sol["time"] = t
            self.exportSolution(i + 1, sol)
            t = t + dt
