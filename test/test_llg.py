import unittest
import os
import shutil
import numpy as np

from numpy.testing import assert_array_almost_equal

from lys_fem import geometry, mf
from lys_fem.fem import FEMProject, DirichletBoundary, InitialCondition, CGSolver, TimeDependentSolver, BackwardEulerSolver, GMRESSolver, FEMSolution
from lys_fem.models import llg


class testProblems_test(unittest.TestCase):
    path = "test/run"

    def setUp(self):
        os.makedirs(self.path, exist_ok=True)
        self._cwd = os.getcwd()
        os.chdir(self.path)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.path)

    def test_nonlinear(self):
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 1, 1))
        p.geometries.add(geometry.Box(1, 0, 0, 1, 1, 1))
        #p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        #p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))
        # p.mesher.setRefinement(1)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(InitialCondition("Initial condition1", [1,0,0], [1]))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver([model], [BackwardEulerSolver(GMRESSolver())], 0.0001, 0.02)
        p.solvers.append(solver)

        # solve
        mf.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("x", data_number=1)
        for w in res:
            assert_array_almost_equal(w.data, np.sqrt(2 * w.x[:, 0]), decimal=2)
