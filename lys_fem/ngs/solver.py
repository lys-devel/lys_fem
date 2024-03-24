import os
import shutil
import numpy as np

from . import mpi, mesh, time
from ngsolve import sqrt


def generateSolver(fem, mesh, model):
    solvers = {"Stationary Solver": StationarySolver, "Relaxation Solver": RelaxationSolver, "Time Dependent Solver": TimeDependentSolver}
    result = []
    for i, s in enumerate(fem.solvers):
        sol = solvers[s.className]
        result.append(sol(s, mesh, model, "Solver" + str(i)))
    return result


class SolverBase:
    def __init__(self, obj, mesh, model, dirname):
        self._obj = obj
        self._mesh = mesh
        self._solver = _NewtonSolver()
        self._integ = time.BackwardEuler(model)
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
    def integrator(self):
        return self._integ

    @property    
    def model(self):
        return self._model

    @property
    def name(self):
        return self._obj.className


class StationarySolver(SolverBase):
    def __init__(self, obj, mesh, model, dirname):
        super().__init__(obj, mesh, model, dirname)
        self._mesh = mesh

    def execute(self):
        self.exportMesh(self._mesh)
        self.exportSolution(0, self.model.solution)

        self.integrator.solve(self._solver)
        self.exportSolution(1, self.model.solution)


class RelaxationSolver(SolverBase):
    def __init__(self, obj, mesh, model, dirname):
        super().__init__(obj, mesh, model, dirname)
        self._tSolver = obj
        self._mesh = mesh

    def execute(self):
        self.exportMesh(self._mesh)
        self.exportSolution(0, self.model.solution)

        t = 0
        dt, dx = self._tSolver.dt0, self._tSolver.dx
        for i in range(1,100):
            self.integrator.solve(self._solver, 1/dt)
            t = t + dt
            print("Step", i, ", t = {:3e}".format(t), ", dt = {:3e}".format(dt), ", dx = {:3e}".format(self._solver.dx))
            self.exportSolution(i, self.model.solution)
            if dt == np.inf:
                break
            dt *= np.sqrt(dx/self._solver.dx)
            if dt > self._tSolver.dt0*1e5:
                dt = np.inf


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
            print("Timestep", i, ", t = {:3e}".format(t), ", dt = {:3e}".format(dt), ", dx = {:3e}".format(self._solver.dx))
            self.integrator.solve(self._solver, 1/dt)
            self.exportSolution(i + 1, self.model.solution)
            t = t + dt


class _NewtonSolver:
    def __init__(self, max_iter=30):
        self._max_iter = max_iter
        self._dx = 0

    def solve(self, F, x, eps=1e-5):
        if not F.isNonlinear:
            self._max_iter=1
        dx = x.vec.CreateVector()
        x0 = x.vec.CreateVector()
        x0.data = x.vec
        for i in range(self._max_iter):
            Fx = F(x.vec)
            dx.data = F.Jacobian(x.vec)*Fx
            x.vec.data -= dx
            R = sqrt(dx.InnerProduct(dx)/x.vec.InnerProduct(x.vec))
            if R < eps:
                print("[Newton solver] Converged in", i, "steps.")
                self._setDifference(x.vec, x0)
                return x
        if self._max_iter !=1:
            print("[Newton solver] NOT Converged in", i, "steps.")
        self._setDifference(x.vec, x0)
        return x

    def _setDifference(self, x, x0):
        x0.data = x0.data - x.data
        self._dx = x0.InnerProduct(x0)/x.InnerProduct(x)

    @property
    def dx(self):
        return self._dx