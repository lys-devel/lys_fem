from lys_fem import geometry
from lys_fem.fem import FEMProject, Material, InitialCondition, StationarySolver, FEMSolution
from lys_fem.models import elasticity

from ..base import FEMTestCase

class elasticity_test(FEMTestCase):
    def dirichlet_1d(self, lib):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # material
        param = elasticity.ElasticParameters()
        mat1 = Material("Material1", [1, 2], [param])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = elasticity.ElasticModel(1)
        model.boundaryConditions.append(elasticity.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(InitialCondition("Initial condition1", 0, [1]))
        model.initialConditions.append(InitialCondition("Initial condition2", 2, [2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("u", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

    def dirichlet_2d(self, lib):
        p = FEMProject(2)

        # geometry
        p.geometries.add(geometry.Rect(0, 0, 0, 1, 1))
        p.geometries.add(geometry.Rect(1, 0, 0, 1, 1))

        # material
        param = elasticity.ElasticParameters()
        mat1 = Material("Material1", [1, 2], [param])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = elasticity.ElasticModel(2)
        model.boundaryConditions.append(elasticity.DirichletBoundary([True, False], geometries=[4, 6]))
        model.initialConditions.append(InitialCondition("Initial condition1", [0, 0], [1]))
        model.initialConditions.append(InitialCondition("Initial condition2", [2, 0], [2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("ux", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

    def dirichlet_3d(self, lib):
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 1, 1))
        p.geometries.add(geometry.Box(1, 0, 0, 1, 1, 1))

        # material
        param = elasticity.ElasticParameters()
        mat1 = Material("Material1", [1, 2], [param])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = elasticity.ElasticModel(3)
        model.boundaryConditions.append(elasticity.DirichletBoundary([True, False, False], geometries=[1, 7]))
        model.initialConditions.append(InitialCondition("Initial condition1", [0, 0, 0], [1]))
        model.initialConditions.append(InitialCondition("Initial condition2", [2, 0, 0], [2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("ux", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])
