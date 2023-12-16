import numpy as np

from numpy.testing import assert_array_almost_equal

from lys_fem import geometry
from lys_fem.fem import FEMProject, DirichletBoundary, InitialCondition, StationarySolver, CGSolver, GMRESSolver, FEMSolution
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
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [1, 3]))
        model.initialConditions.append(InitialCondition("Initial condition1", 0.0, [1]))
        model.initialConditions.append(InitialCondition("Initial condition2", 2.0, [2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver(CGSolver(), [model])
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("x", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])