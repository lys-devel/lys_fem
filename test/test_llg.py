import numpy as np

from numpy.testing import assert_array_almost_equal

from lys_fem import geometry, mf
from lys_fem.fem import FEMProject, DirichletBoundary, InitialCondition, CGSolver, TimeDependentSolver, BackwardEulerSolver, GMRESSolver, FEMSolution
from lys_fem.models import llg

from .base import FEMTestCase

class testProblems_test(FEMTestCase):
    def test_llg(self):
        return
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 1, 1))
        p.geometries.add(geometry.Box(1, 0, 0, 1, 1, 1))
        # p.mesher.setRefinement(1)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(InitialCondition("Initial condition1", [1, 0, 0], [1]))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver([model], [BackwardEulerSolver(GMRESSolver())], 0.1, 3)
        p.solvers.append(solver)

        # solve
        mf.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("x", data_number=1)
        for w in res:
            assert_array_almost_equal(w.data, np.sqrt(2 * w.x[:, 0]), decimal=2)
