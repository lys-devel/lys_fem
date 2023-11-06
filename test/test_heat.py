import unittest
import os
import shutil

from numpy.testing import assert_array_almost_equal

from lys_fem import geometry, mf
from lys_fem.fem import FEMProject, Material, DirichletBoundary, NeumannBoundary, InitialCondition, StationarySolver, CGSolver, FEMSolution
from lys_fem.models import heat


class elasticity_test(unittest.TestCase):
    path = "test/run"

    def setUp(self):
        os.makedirs(self.path, exist_ok=True)
        self._cwd = os.getcwd()
        os.chdir(self.path)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.path)

    def test_1d_dirichlet(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # material
        param = heat.HeatConductionParameters()
        mat1 = Material("Material1", [1, 2], [param])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = heat.HeatConductionModel()
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [1, 3]))
        model.initialConditions.append(InitialCondition("Initial condition1", [0], [1]))
        model.initialConditions.append(InitialCondition("Initial condition2", [2], [2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver([CGSolver(model)])
        p.solvers.append(stationary)

        # solve
        mf.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("T", data_number=1)
        for w in res:
            assert_array_almost_equal(w.data, w.x[:, 0])

    def test_1d_neumann(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # material
        param = heat.HeatConductionParameters()
        mat1 = Material("Material1", [1, 2], [param])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = heat.HeatConductionModel()
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [1]))
        model.boundaryConditions.append(NeumannBoundary("Neumann boundary1", [0.5], [3]))
        model.initialConditions.append(InitialCondition("Initial condition1", [0], [1, 2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver([CGSolver(model)])
        p.solvers.append(stationary)

        # solve
        mf.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("T", data_number=1)
        for w in res:
            assert_array_almost_equal(w.data, w.x[:, 0] / 2)
