import unittest
import os
import shutil
import numpy as np

from numpy.testing import assert_array_almost_equal

from lys_fem import geometry, mf
from lys_fem.fem import FEMProject, DirichletBoundary, InitialCondition, StationarySolver, CGSolver, GMRESSolver, FEMSolution
from lys_fem.models import test


class testProblems_test(unittest.TestCase):
    path = "test/run"

    def setUp(self):
        os.makedirs(self.path, exist_ok=True)
        self._cwd = os.getcwd()
        os.chdir(self.path)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.path)

    def test_linear(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # model: boundary and initial conditions
        model = test.LinearTestModel()
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [1, 3]))
        model.initialConditions.append(InitialCondition("Initial condition1", [0], [1]))
        model.initialConditions.append(InitialCondition("Initial condition2", [2], [2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver([model], [CGSolver])
        p.solvers.append(stationary)

        # solve
        mf.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("x", data_number=1)
        for w in res:
            assert_array_almost_equal(w.data, w.x[:, 0])

    def test_nonlinear(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))
        p.mesher.setRefinement(1)

        # model: boundary and initial conditions
        model = test.NonlinearTestModel()
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [1, 3]))
        model.initialConditions.append(InitialCondition("Initial condition1", ["x"], [1]))
        model.initialConditions.append(InitialCondition("Initial condition2", ["x"], [2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver([model], [GMRESSolver])
        p.solvers.append(stationary)

        # solve
        mf.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("x", data_number=1)
        for w in res:
            assert_array_almost_equal(w.data, np.sqrt(2 * w.x[:, 0]), decimal=2)
