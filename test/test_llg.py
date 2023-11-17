import numpy as np

from lys_fem import geometry, mf
from lys_fem.fem import FEMProject, DirichletBoundary, InitialCondition, CGSolver, TimeDependentSolver, StationarySolver, BackwardEulerSolver, GMRESSolver, FEMSolution
from lys_fem.models import llg

from .base import FEMTestCase

T = 2*np.pi/1.760859770e11

class testLLG_test(FEMTestCase):
    def test_precession(self):
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 1, 1))

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(InitialCondition("Initial condition1", [1, 0, 0], [1]))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver([model], [BackwardEulerSolver(GMRESSolver())], T/100, T/2)
        p.solvers.append(solver)

        # solve
        mf.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("mx", data_number=25)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)
        res = sol.eval("my", data_number=25)
        for w in res:
            self.assert_array_almost_equal(w.data, np.ones(w.data.shape), decimal=3)
        res = sol.eval("mx", data_number=50)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.ones(w.data.shape), decimal=3)
        res = sol.eval("my", data_number=50)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)
