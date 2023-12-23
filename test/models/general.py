import sympy as sp
import numpy as np

from lys_fem import geometry
from lys_fem.fem import FEMProject, Source, DirichletBoundary, NeumannBoundary, InitialCondition, FEMSolution
from lys_fem.fem import StationarySolver, TimeDependentSolver
from lys_fem.models import general

from ..base import FEMTestCase

class poisson_test(FEMTestCase):
    def dirichlet_1d(self, lib):
        p = FEMProject(1)

        r0 = 3
        a = 1
        rho = 1
        # geometry
        p.geometries.add(geometry.Line(-r0, 0, 0, -a, 0, 0))
        p.geometries.add(geometry.Line(-a, 0, 0, a, 0, 0))
        p.geometries.add(geometry.Line(a, 0, 0, r0, 0, 0))
        p.mesher.setRefinement(1)

        # model: boundary and initial conditions
        model = general.PoissonModel()
        model.domainConditions.append(Source("Source1", rho, [2]))
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [1, 4]))
        model.initialConditions.append(InitialCondition("Initial condition1", 0, [1, 3]))
        model.initialConditions.append(InitialCondition("Initial condition2", 1, [2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        def solution(r):
            return rho*np.where(r<=-1, -(r+3), np.where(r>=1, r-3, r**2/2-2.5))


        sol = FEMSolution(".", p)
        res = sol.eval("phi", data_number=1)
        for w in res:
            r = w.x[:,0]
            self.assert_allclose(w.data, solution(r), atol=1e-6, rtol=0)
       
    def dirichlet_2d(self, lib):
        p = FEMProject(2)

        r0 = 3
        a = 1
        rho = 1
        # geometry
        p.geometries.add(geometry.Disk(0, 0, 0, a, a))
        p.geometries.add(geometry.Disk(0, 0, 0, r0, r0))
        p.mesher.setRefinement(4)

        # model: boundary and initial conditions
        model = general.PoissonModel()
        model.domainConditions.append(Source("Source1", rho, [1]))
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [1]))
        model.initialConditions.append(InitialCondition("Initial condition1", 0, [2]))
        model.initialConditions.append(InitialCondition("Initial condition2", 1, [1]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        def solution(r):
            return rho/4*np.where(r<=1, r**2-1-2*np.log(r0), 2*np.log(r/r0))


        sol = FEMSolution(".", p)
        res = sol.eval("phi", data_number=1)
        for w in res:
            r = np.sqrt(w.x[:,0]**2+w.x[:,1]**2)
            self.assert_allclose(w.data, solution(r), atol=1e-3, rtol=0)

    def dirichlet_3d(self, lib):
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Sphere(0, 0, 0, 1))
        p.geometries.add(geometry.Sphere(0, 0, 0, 2))
        p.mesher.setRefinement(1)
        p.mesher.setPartialRefinement(3, 1, 2)

        # model: boundary and initial conditions
        model = general.PoissonModel()
        model.domainConditions.append(Source("Source1", 1, [1]))
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [2]))
        model.initialConditions.append(InitialCondition("Initial condition1", 0, [1,2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        def solution(r, a, r0, rho):
            return np.where(r<=a, rho*(-r**2/6+a**2/2-a**3/(3*r0)), rho*(-a**3/(3*r0)+a**3/(3*r)))


        sol = FEMSolution(".", p)
        res = sol.eval("phi", data_number=1)
        for w in res:
            r = np.sqrt(w.x[:,0]**2+w.x[:,1]**2+w.x[:,2]**2)
            self.assert_allclose(w.data, -solution(r, 1, 2, 1), atol=0.1, rtol=0)

    def infinite_3d(self, lib):
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Sphere(0, 0, 0, 1))
        p.geometries.add(geometry.Box(-0.5, -0.5, -0.5))
        p.geometries.add(geometry.InfiniteVolume(1, 1, 1, 2, 2, 2))
        p.mesher.setRefinement(1)

        # model: boundary and initial conditions
        model = general.PoissonModel()
        model.domainConditions.append(Source("Source1", 1, [1]))
        model.domainConditions.append(general.InfiniteVolume("InfVol", []))
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        def solution(r, a, rho):
            return rho*np.where(r<=a, -r**2/6+a**2/2, a**3/(3*r))


        sol = FEMSolution(".", p)
        res = sol.eval("phi", data_number=1)
        for w in res:
            r = np.sqrt(w.x[:,0]**2+w.x[:,1]**2+w.x[:,2]**2)
            self.assert_allclose(w.data, -solution(r, 1, 1), atol=0.1, rtol=0)
