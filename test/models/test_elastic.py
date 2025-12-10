import sympy as sp
import numpy as np

from lys_fem import geometry
from lys_fem.fem import FEMProject, Material, StationarySolver, TimeDependentSolver, FEMSolution
from lys_fem.models import elasticity, heat

from ..base import FEMTestCase

class elasticity_test(FEMTestCase):
    def test_dirichlet_1d(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # material
        param = elasticity.ElasticParameters()
        mat1 = Material([param], geometries=[1, 2])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = elasticity.ElasticModel(1)
        model.boundaryConditions.append(elasticity.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(elasticity.InitialCondition(0, geometries=[1]))
        model.initialConditions.append(elasticity.InitialCondition(2, geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("u[0]", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

    def test_dirichlet_2d(self):
        p = FEMProject(2)

        # geometry
        p.geometries.add(geometry.Rect(0, 0, 0, 1, 1))
        p.geometries.add(geometry.Rect(1, 0, 0, 1, 1))

        # material
        param = elasticity.ElasticParameters()
        mat1 = Material([param], geometries=[1, 2])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = elasticity.ElasticModel(2)
        model.boundaryConditions.append(elasticity.DirichletBoundary([True, False], geometries=[4, 6]))
        model.initialConditions.append(elasticity.InitialCondition([0, 0], geometries=[1]))
        model.initialConditions.append(elasticity.InitialCondition([2, 0], geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("u[0]", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

    def test_dirichlet_3d(self):
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 1, 1))
        p.geometries.add(geometry.Box(1, 0, 0, 1, 1, 1))

        # material
        param = elasticity.ElasticParameters()
        mat1 = Material([param], geometries=[1, 2])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = elasticity.ElasticModel(3)
        model.boundaryConditions.append(elasticity.DirichletBoundary([True, False, False], geometries=[1, 7]))
        model.initialConditions.append(elasticity.InitialCondition([0, 0, 0], geometries=[1]))
        model.initialConditions.append(elasticity.InitialCondition([2, 0, 0], geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("u[0]", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

    def test_tdep_1d(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 2, 0, 0))
        p.mesher.setRefinement(5)
        
        # material
        param = elasticity.ElasticParameters()
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        x = sp.symbols("x")

        # model: boundary and initial conditions
        model = elasticity.ElasticModel(1)
        model.initialConditions.append(elasticity.InitialCondition(sp.exp(-(x/0.1)**2), geometries=[1]))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(0.005, 1, condensation=True)
        p.solvers.append(solver)

        # solve
        p.run()

        # solution
        sol = FEMSolution()

        res = sol.eval("u[0]", data_number=0)
        for w in res:
            self.assert_array_almost_equal(w.data, np.exp(-((w.x[:, 0])/0.1)**2), decimal=3)
        res = sol.eval("u[0]", data_number=100)
        for w in res:
            self.assert_array_almost_equal(w.data, np.exp(-((w.x[:, 0]-np.sqrt(3)/2)/0.1)**2)/2, decimal=2)

    def test_thermoelasticity_1d(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 0.1e-6, 0, 0))
        p.mesher.setRefinement(4)

        # material
        param = elasticity.ElasticParameters(rho=1000, C=[100e9, 60e9], alpha=np.eye(3)*2e-6, type="isotropic")
        param2 = heat.HeatConductionParameters()
        mat1 = Material([param, param2], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = elasticity.ElasticModel(1)
        model.initialConditions.append(elasticity.InitialCondition(0, geometries="all"))
        model.domainConditions.append(elasticity.ThermoelasticStress("T", geometries="all"))
        p.models.append(model)

        model2 = heat.HeatConductionModel()
        model2.initialConditions.append(heat.InitialCondition(100, geometries="all"))
        p.models.append(model2)

        # solver
        solver = TimeDependentSolver(0.5e-13, 0.5e-13*500)
        p.solvers.append(solver)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        self.assertAlmostEqual(sol.eval("u[0]", data_number=400, coords=0), 0, places=11)

    def test_thermoelasticity_2d(self):
        p = FEMProject(2)

        # geometry
        p.geometries.add(geometry.Rect(0, 0, 0, 0.1e-6, 1e-6))
        p.mesher.setRefinement(2)

        # material
        param = elasticity.ElasticParameters(rho=1000, C=[100e9, 60e9], alpha=np.eye(3)*2e-6, type="isotropic")
        param2 = heat.HeatConductionParameters()
        mat1 = Material([param, param2], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = elasticity.ElasticModel(2)
        model.initialConditions.append(elasticity.InitialCondition([0,0], geometries="all"))
        model.domainConditions.append(elasticity.ThermoelasticStress("T", geometries="all"))
        p.models.append(model)

        model2 = heat.HeatConductionModel()
        model2.initialConditions.append(heat.InitialCondition(100, geometries="all"))
        p.models.append(model2)

        # solver
        solver = TimeDependentSolver(1e-13, 1e-13*50)
        p.solvers.append(solver)

        # solve
        p.run()

    def test_thermoelasticity_time(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 0.1e-6, 0, 0))
        p.mesher.setRefinement(4)

        # material
        param = elasticity.ElasticParameters(rho=1000, C=[100e9, 60e9], alpha=np.eye(3)*2e-6, type="isotropic")
        param2 = elasticity.UserDefinedParameters(T=100)
        mat = Material([param, param2], geometries="all")
        p.materials.append(mat)

        # model: boundary and initial conditions
        model = elasticity.ElasticModel(1)
        model.initialConditions.append(elasticity.InitialCondition(0, geometries="all"))
        model.domainConditions.append(elasticity.ThermoelasticStress("T*step(t-1e-12)", geometries="all"))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(0.5e-13, 0.5e-13*500)
        p.solvers.append(solver)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        self.assertAlmostEqual(sol.eval("u[0]", data_number=420, coords=0), 0, places=11)

    def test_rotation_1d(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 2, 0, 0))
        p.mesher.setRefinement(5)
        
        # material
        C = np.eye(6)
        C[0,0] = 1/4
        C[1,1] = 1/2
        C[2,2] = 1
        R = [[0,1,0], [0,0,1], [1,0,0]]
        param = elasticity.ElasticParameters(C=C, type="triclinic")
        mat1 = Material([param], geometries="all", coord=R)
        p.materials.append(mat1)

        x = sp.symbols("x")

        # model: boundary and initial conditions
        model = elasticity.ElasticModel(1)
        model.initialConditions.append(elasticity.InitialCondition(sp.exp(-(x/0.1)**2), geometries=[1]))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(0.0025, 1, condensation=True)
        p.solvers.append(solver)

        # solve
        p.run()

        # solution
        sol = FEMSolution()

        res = sol.eval("u[0]", data_number=0)
        for w in res:
            self.assert_array_almost_equal(w.data, np.exp(-((w.x[:, 0])/0.1)**2), decimal=3)
        res = sol.eval("u[0]", data_number=-1)
        for w in res:
            self.assert_array_almost_equal(w.data, np.exp(-((w.x[:, 0]-1)/0.1)**2)/2, decimal=2)