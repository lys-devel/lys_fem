import numpy as np
import sympy as sp

from numpy.testing import assert_array_almost_equal

from lys_fem import geometry
from lys_fem.fem import FEMProject, StationarySolver, TimeDependentSolver, FEMSolution, SolverStep
from lys_fem.models import test

from ..base import FEMTestCase

class testProblems_test(FEMTestCase):
    def linear(self, lib):
        a = sp.symbols("a")

        p = FEMProject(1)
        p.parameters[a] = 1

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, a, 0, 0))
        p.geometries.add(geometry.Line(a, 0, 0, 2, 0, 0))

        # model: boundary and initial conditions
        model = test.LinearTestModel()
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(test.InitialCondition(0.0, geometries=[1]))
        model.initialConditions.append(test.InitialCondition(2.0, geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution()
        res = sol.eval("x", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

    def nonlinear(self, lib):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))
        p.mesher.setRefinement(2)

        # model: boundary and initial conditions
        x = sp.Symbol("x")
        model = test.NonlinearTestModel()
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(test.InitialCondition(x, geometries=[1]))
        model.initialConditions.append(test.InitialCondition(x, geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution()
        res = sol.eval("x", data_number=1)
        for w in res:
            assert_array_almost_equal(w.data, np.sqrt(2 * w.x[:, 0]), decimal=2)
        c = np.array([0.5,0.6,0.7])
        assert_array_almost_equal(sol.eval("x", data_number=1, coords=c), np.sqrt(2*c))

    def twoVars1(self, lib):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))

        # model: boundary and initial conditions
        model = test.TwoVariableTestModel()
        model.initialConditions.append(test.InitialCondition([1,0], geometries="all"))
        p.models.append(model)

        # solver
        stationary = TimeDependentSolver(0.001, 0.1)
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution()
        t = np.linspace(0,0.1,101)
        x = [sol.eval("x", coords=0, data_number=i) for i in range(101)]
        y = [sol.eval("y", coords=0, data_number=i) for i in range(101)]
        self.assert_array_almost_equal(x, (np.exp(-2*t)+1)/2, decimal=4)
        self.assert_array_almost_equal(y, (1-np.exp(-2*t))/2, decimal=4)

    def twoVars_step(self, lib):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))

        # model: boundary and initial conditions
        model = test.TwoVariableTestModel()
        model.initialConditions.append(test.InitialCondition([1,0], geometries="all"))
        p.models.append(model)

        # solver
        steps = [SolverStep(["x"]), SolverStep(["y"])]
        stationary = TimeDependentSolver(0.001, 0.1, steps=steps)
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution()
        t = np.linspace(0,0.1,101)
        x = [sol.eval("x", coords=0, data_number=i) for i in range(101)]
        y = [sol.eval("y", coords=0, data_number=i) for i in range(101)]
        self.assert_array_almost_equal(x, (np.exp(-2*t)+1)/2, decimal=4)
        self.assert_array_almost_equal(y, (1-np.exp(-2*t))/2, decimal=4)