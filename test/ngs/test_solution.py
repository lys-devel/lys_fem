from lys_fem import geometry, ngs
from lys_fem.fem import FEMProject, StationarySolver, TimeDependentSolver, FEMSolution, SolverStep, Material
from lys_fem.models import test

from ..base import FEMTestCase


class solution_test(FEMTestCase):
    def test_solution(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # model: boundary and initial conditions
        model = test.LinearTestModel()
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(test.InitialCondition(0.0, geometries=[1]))
        model.initialConditions.append(test.InitialCondition(2.0, geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver(steps=[SolverStep()])
        p.solvers.append(stationary)

        # solve
        ngs.run(p)

        # solution
        sol = FEMSolution()

        # mesh
        res = sol.eval("X", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

        # 1d array
        res = sol.eval("X", data_number=1, coords=[0,1,2])
        self.assert_array_almost_equal(res, [0,1,2])

        # multi dimensional array
        res = sol.eval("X", data_number=1, coords=[[0,1,2], [2,1,0]])
        self.assert_array_almost_equal(res, [[0,1,2], [2,1,0]])
