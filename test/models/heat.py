from scipy import special
import numpy as np

from lys_fem import geometry
from lys_fem.fem import FEMProject, Material, FEMSolution, StationarySolver, TimeDependentSolver
from lys_fem.models import heat
from ..base import FEMTestCase

class heat_test(FEMTestCase):     
    def dirichlet_1d(self, lib):
        p = FEMProject(1)

        # geometry
        p.geometries.scale = 100
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))
        p.mesher.setRefinement(5)

        # material
        param = heat.HeatConductionParameters()
        mat1 = Material([param], geometries=[1, 2])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = heat.HeatConductionModel()
        model.boundaryConditions.append(heat.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(heat.InitialCondition(0, geometries=[1]))
        model.initialConditions.append(heat.InitialCondition(2, geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution()
        res = sol.eval("T", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

    def neumann_1d(self, lib):
        p = FEMProject(1)

        # geometry
        p.geometries.scale = 100
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # material
        param = heat.HeatConductionParameters()
        mat1 = Material([param], geometries=[1, 2])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = heat.HeatConductionModel()
        model.boundaryConditions.append(heat.DirichletBoundary([True], geometries=[1]))
        model.boundaryConditions.append(heat.NeumannBoundary(0.5, geometries=[3]))
        model.initialConditions.append(heat.InitialCondition(0, geometries=[1, 2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution()
        res = sol.eval("T", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0] / 2)

    def tdep_1d(self, lib):
        def calc_temp(x, t, kappa=1, DT=1, T_0=0):
            return T_0 + DT * special.erfc(-x / np.sqrt(4 * kappa * t)) / 2

        p = FEMProject(1)

        # geometry
        p.geometries.scale=100
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # material
        param = heat.HeatConductionParameters()
        mat1 = Material([param], geometries=[1, 2])
        p.materials.append(mat1)

        p.mesher.setRefinement(6)

        # model: boundary and initial conditions
        model = heat.HeatConductionModel()
        model.initialConditions.append(heat.InitialCondition(0, geometries=[1]))
        model.initialConditions.append(heat.InitialCondition(1, geometries=[2]))
        p.models.append(model)

        # solver
        stationary = TimeDependentSolver(0.0001, 0.02)
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution()

        res = sol.eval("T", data_number=0)
        for w in res:
            self.assert_allclose(w.data, np.heaviside(w.x[:, 0] - 1, 0.5), rtol=0, atol=0.001)

        res = sol.eval("T", data_number=100)
        for w in res:
            self.assert_allclose(w.data, calc_temp(w.x[:, 0] - 1, 0.01), rtol=0, atol=0.001)

    def tdep_bdf2(self, lib):
        def calc_temp(x, t, kappa=1, DT=1, T_0=0):
            return T_0 + DT * special.erfc(-x / np.sqrt(4 * kappa * t)) / 2

        p = FEMProject(1)

        # geometry
        p.geometries.scale=100
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # material
        param = heat.HeatConductionParameters()
        mat1 = Material([param], geometries=[1, 2])
        p.materials.append(mat1)

        p.mesher.setRefinement(6)

        # model: boundary and initial conditions
        model = heat.HeatConductionModel()
        model.initialConditions.append(heat.InitialCondition(0, geometries=[1]))
        model.initialConditions.append(heat.InitialCondition(1, geometries=[2]))
        p.models.append(model)

        # solver
        stationary = TimeDependentSolver(0.0001, 0.02, method="BDF2")
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution()

        res = sol.eval("T", data_number=0)
        for w in res:
            self.assert_allclose(w.data, np.heaviside(w.x[:, 0] - 1, 0.5), rtol=0, atol=0.001)

        res = sol.eval("T", data_number=100)
        for w in res:
            self.assert_allclose(w.data, calc_temp(w.x[:, 0] - 1, 0.01), rtol=0, atol=0.001)