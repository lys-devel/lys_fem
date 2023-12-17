import numpy as np

from lys_fem import geometry
from lys_fem.fem import FEMProject, Source, DirichletBoundary, NeumannBoundary, InitialCondition, FEMSolution
from lys_fem.fem import StationarySolver, TimeDependentSolver
from lys_fem.models import general

from ..base import FEMTestCase

class poisson_test(FEMTestCase):
    def dirichlet_3d(self, lib):
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Sphere(0, 0, 0, 2))
        p.geometries.add(geometry.Sphere(0, 0, 0, 5))
        p.mesher.setRefinement(2)

        # model: boundary and initial conditions
        model = general.PoissonModel()
        model.domainConditions.append(Source("Source1", 3, [1]))
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [2]))
        model.initialConditions.append(InitialCondition("Initial condition1", 0, [1,2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver([model])
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        def solution(r, a, r0, rho):
            return np.where(r<=a, rho*(-r**2/6+a**2/2-a**3/(3*r0)), rho*(-a**3/(3*r0)+a**3/(3*r)))
                
        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("phi", data_number=1)
        for w in res:
            r = np.sqrt(w.x[:,0]**2+w.x[:,1]**2+w.x[:,2]**2)
            self.assert_array_almost_equal(w.data, solution(r, 2, 5, 3))

