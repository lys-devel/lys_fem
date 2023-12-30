import os
import shutil

from . import mpi, mesh
from ngsolve import sqrt


def generateSolver(fem, mesh, model):
    solvers = {"Stationary Solver": StationarySolver, "Time Dependent Solver": TimeDependentSolver}
    result = []
    for i, s in enumerate(fem.solvers):
        sol = solvers[s.name]
        result.append(sol(s, mesh, model, "Solver" + str(i)))
    return result


class SolverBase:
    def __init__(self, obj, mesh, model, dirname):
        self._obj = obj
        self._mesh = mesh
        self._solver = _NewtonSolver()
        self._model = model
        self._prepareDirectory(dirname)

    def _prepareDirectory(self, dirname):
        self._dirname = "Solutions/" + dirname
        if mpi.isRoot:
            if os.path.exists(self._dirname):
                shutil.rmtree(self._dirname)
        os.makedirs(self._dirname, exist_ok=True)

    def exportMesh(self, m):
        mesh.exportMesh(m, self._dirname + "/mesh.npz")

    def exportSolution(self, index, solution):
        mesh.exportSolution(self._mesh, solution, self._dirname + "/data" + str(index))

    @property
    def solver(self):
        return self._solver

    @property    
    def model(self):
        return self._model

    @property
    def name(self):
        return self._obj.name


class StationarySolver(SolverBase):
    def __init__(self, obj, mesh, model, dirname):
        super().__init__(obj, mesh, model, dirname)
        self._mesh = mesh

    def execute(self):
        self.exportMesh(self._mesh)
        self.exportSolution(0, self.model.solution)

        sol = self.model.solve(self._solver)
        self.exportSolution(1, sol)


class TimeDependentSolver(SolverBase):
    def __init__(self, obj, mesh, model, dirname):
        super().__init__(obj, mesh, model, dirname)
        self._tSolver = obj
        self._mesh = mesh

    def execute(self):
        self.exportMesh(self._mesh)
        self.exportSolution(0, self.model.solution)

        t = 0
        for i, dt in enumerate(self._tSolver.getStepList()):
            print("timestep", i, ", t =", t)
            sol = self.model.solve(self._solver, 1/dt)
            self.exportSolution(i + 1, sol)
            t = t + dt


class _NewtonSolver:
    def __init__(self, max_iter=30):
        self._max_iter = max_iter

    def solve(self, F, x, eps=1e-10):
        if not F.isNonlinear:
            self._max_iter=1
        dx = x.vec.CreateVector()
        for i in range(self._max_iter):
            Fx = F(x)
            dx.data = F.Jacobian(x) * Fx
            x.vec.data -= dx
            R = sqrt(abs(Fx.InnerProduct(dx)))
            if R < eps:
                print("Newton", i)
                return x
        if self._max_iter !=1:
            print("Newton solver does not converge.")
        return x


