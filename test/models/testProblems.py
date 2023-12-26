import numpy as np
import sympy as sp

from numpy.testing import assert_array_almost_equal

from lys_fem import geometry
from lys_fem.fem import FEMProject, InitialCondition, StationarySolver, FEMSolution
from lys_fem.models import test

from ..base import FEMTestCase

class testProblems_test(FEMTestCase):
    def linear(self, lib):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # model: boundary and initial conditions
        model = test.LinearTestModel()
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(InitialCondition("Initial condition1", 0.0, [1]))
        model.initialConditions.append(InitialCondition("Initial condition2", 2.0, [2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution(".", p)
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
        model.initialConditions.append(InitialCondition("Initial condition1", x, [1]))
        model.initialConditions.append(InitialCondition("Initial condition2", x, [2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("x", data_number=1)
        for w in res:
            assert_array_almost_equal(w.data, np.sqrt(2 * w.x[:, 0]), decimal=2)
