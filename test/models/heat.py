from scipy import special
import numpy as np
import sympy as sp

from lys_fem import geometry, ngs
from lys_fem.fem import FEMProject, Material, DirichletBoundary, NeumannBoundary, InitialCondition, FEMSolution
from lys_fem.fem import StationarySolver, CGSolver, TimeDependentSolver
from lys_fem.models import heat

from ..base import FEMTestCase

x = sp.Symbol("x")

class heat_test(FEMTestCase):     
    def dirichlet_1d(self, lib):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))
        p.mesher.setRefinement(5)

        # material
        param = heat.HeatConductionParameters()
        mat1 = Material("Material1", [1, 2], [param])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = heat.HeatConductionModel()
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [1, 3]))
        model.initialConditions.append(InitialCondition("Initial condition1", 0, [1]))
        model.initialConditions.append(InitialCondition("Initial condition2", 2, [2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver(CGSolver(), [model])
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("T", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

    def neumann_1d(self, lib):
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
        model.boundaryConditions.append(NeumannBoundary("Neumann boundary1", 0.5, [3]))
        model.initialConditions.append(InitialCondition("Initial condition1", 0, [1, 2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver(CGSolver(), [model])
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("T", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0] / 2)