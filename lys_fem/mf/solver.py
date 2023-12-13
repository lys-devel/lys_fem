import os
import shutil
import numpy as np

from . import mfem
from .models import CompositeModel


def generateSolver(fem, mesh, models):
    solvers = {"Stationary Solver": StationarySolver, "Time Dependent Solver": TimeDependentSolver}
    result = []
    for i, s in enumerate(fem.solvers):
        sol = solvers[s.name]
        model_list = [models[fem.models.index(m)] for m in s.models]
        result.append(sol(s, mesh, model_list, "Solver" + str(i)))
    return result


class SolverBase:
    def __init__(self, obj, dirname):
        self._obj = obj
        self._solver = self._createFEMSolver(obj.solver)
        self._prepareDirectory(dirname)

    def _prepareDirectory(self, dirname):
        self._dirname = "Solutions/" + dirname
        if mfem.isRoot:
            if os.path.exists(self._dirname):
                shutil.rmtree(self._dirname)
        os.makedirs(self._dirname, exist_ok=True)

    def _createFEMSolver(self, sol):
        subSolvers = {"CG Solver": CGSolver, "GMRES Solver": GMRESSolver}
        return NewtonSolver(subSolvers[sol.name](sol))

    def exportMesh(self, mesh):
        coords, elems = mfem.getMesh(mesh)
        if mfem.isRoot:
            np.savez(self._dirname + "/mesh.npz", coords = coords, mesh=elems)

    def exportSolution(self, index, solution):
        if mfem.isRoot:
            np.savez(self._dirname + "/data" + str(index), **solution)

    @property
    def solver(self):
        return self._solver

    @property
    def name(self):
        return self._obj.name


class StationarySolver(SolverBase):
    def __init__(self, obj, mesh, models, dirname):
        super().__init__(obj, dirname)
        self._mesh = mesh
        self._model = CompositeModel(mesh, models, "Stationary")

    def execute(self):
        self.exportMesh(self._mesh)
        self.exportSolution(0, self._model.solution)

        sol = self._model.solve(self.solver)
        self.exportSolution(1, sol)
        mfem.print_("Stationary problem has been solved")


class TimeDependentSolver(SolverBase):
    def __init__(self, obj, mesh, models, dirname):
        super().__init__(obj, dirname)
        self._tSolver = obj
        self._mesh = mesh
        self._model = CompositeModel(mesh, models, "TimeDependent")

    def execute(self):
        self.exportMesh(self._mesh)
        self.exportSolution(0, self._model.solution)
        t = 0
        for i, dt in enumerate(self._tSolver.getStepList()):
            mfem.print_("timestep", i, ", t =", t)
            sol = self._model.solve(self.solver, dt)
            self.exportSolution(i + 1, sol)
            t = t + dt


class CGSolver(mfem.CGSolver):
    def __init__(self, sol):
        super().__init__()


class GMRESSolver(mfem.GMRESSolver):
    def __init__(self, sol):
        super().__init__()


class NewtonSolver:
    def __init__(self, subSolver, max_iter=30):
        self._solver = subSolver
        self._max_iter = max_iter

    def setMaxIteration(self, max_iter):
        self._max_iter = max_iter

    def setPreconditioner(self, prec):
        self._solver.SetPreconditioner(prec)

    def solve(self, F, x, eps=1e-7):
        x = mfem.Vector(x)
        Ji = self._solver
        for i in range(self._max_iter):
            F.update(x)
            J = F.grad(x)
            Ji.SetOperator(J)
            dx = Ji * F(x)
            x -= dx
            norm = mfem.getMax(x.Norml2())
            R = mfem.getMax(dx.Norml2())
            if norm != 0:
                R = R/norm
            if R < eps:
                print("Newton", i)
                return x
        if self._max_iter !=1:
            print("Newton solver does not converge.")
        return x


