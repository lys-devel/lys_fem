from scipy import special
import numpy as np

from numpy.testing import assert_array_almost_equal, assert_allclose

from lys_fem import geometry, mf
from lys_fem.fem import FEMProject, Material, DirichletBoundary, NeumannBoundary, InitialCondition, FEMSolution
from lys_fem.fem import StationarySolver, CGSolver, TimeDependentSolver, BackwardEulerSolver
from lys_fem.models import heat

from .base import FEMTestCase

class elasticity_test(FEMTestCase):
    def test_1d_dirichlet(self):
        p = self.__create1D()

        # model: boundary and initial conditions
        model = heat.HeatConductionModel()
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
        res = sol.eval("T", data_number=1)
        for w in res:
            assert_array_almost_equal(w.data, w.x[:, 0])

    def test_1d_neumann(self):
        p = self.__create1D()

        # model: boundary and initial conditions
        model = heat.HeatConductionModel()
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [1]))
        model.boundaryConditions.append(NeumannBoundary("Neumann boundary1", [0.5], [3]))
        model.initialConditions.append(InitialCondition("Initial condition1", [0], [1, 2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver([model], [CGSolver()])
        p.solvers.append(stationary)

        # solve
        mf.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("T", data_number=1)
        for w in res:
            assert_array_almost_equal(w.data, w.x[:, 0] / 2)

    def test_1d_tdep(self):
        def calc_temp(x, t, kappa=1, DT=1, T_0=0):
            return T_0 + DT * special.erfc(-x / np.sqrt(4 * kappa * t)) / 2

        p = self.__create1D()
        p.mesher.setRefinement(6)

        # model: boundary and initial conditions
        model = heat.HeatConductionModel()
        model.initialConditions.append(InitialCondition("Initial condition1", ["heaviside(x-1,0.5)"], [1, 2]))
        p.models.append(model)

        # solver
        stationary = TimeDependentSolver([model], [BackwardEulerSolver(CGSolver())], 0.0001, 0.02)
        p.solvers.append(stationary)

        # solve
        mf.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("T", data_number=100)

        for w in res:
            assert_allclose(w.data, calc_temp(w.x[:, 0] - 1, 0.01), rtol=0, atol=0.001)

    def test_3d_tdep(self):
        return

        def calc_temp(x, t, kappa=1, DT=1, T_0=0):
            return T_0 + DT * special.erfc(-x / np.sqrt(4 * kappa * t)) / 2

        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 1, 1))
        p.geometries.add(geometry.Box(1, 0, 0, 1, 1, 1))

        # material
        param = heat.HeatConductionParameters()
        mat1 = Material("Material1", [1, 2], [param])
        p.materials.append(mat1)
        p.mesher.setRefinement(2)

        # model: boundary and initial conditions
        model = heat.HeatConductionModel()
        model.initialConditions.append(InitialCondition("Initial condition1", ["heaviside(x-1,0.5)"], [1, 2]))
        p.models.append(model)

        # solver
        stationary = TimeDependentSolver([model], [BackwardEulerSolver(CGSolver())], 0.0001, 0.02)
        p.solvers.append(stationary)

        import time
        start = time.time()
        # solve
        mf.run(p)
        print(time.time() - start)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("T", data_number=100)

        for w in res:
            assert_allclose(w.data, calc_temp(w.x[:, 0] - 1, 0.01), rtol=0, atol=0.02)

    def __create1D(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # material
        param = heat.HeatConductionParameters()
        mat1 = Material("Material1", [1, 2], [param])
        p.materials.append(mat1)

        return p
