
import numpy as np

from lys_fem import geometry
from lys_fem.fem import FEMProject, FEMSolution, Material, StationarySolver
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
        model.domainConditions.append(general.Source(rho, geometries=[2]))
        model.boundaryConditions.append(general.DirichletBoundary([True], geometries=[1, 4]))
        model.initialConditions.append(general.InitialCondition(0, geometries=[1, 3]))
        model.initialConditions.append(general.InitialCondition(1, geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        def solution(r):
            return rho*np.where(r<=-1, -(r+3), np.where(r>=1, r-3, r**2/2-2.5))


        sol = FEMSolution()
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
        model.domainConditions.append(general.Source(rho, geometries=[1]))
        model.boundaryConditions.append(general.DirichletBoundary([True], geometries=[1]))
        model.initialConditions.append(general.InitialCondition(0, geometries=[2]))
        model.initialConditions.append(general.InitialCondition(1, geometries=[1]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        def solution(r):
            return rho/4*np.where(r<=1, r**2-1-2*np.log(r0), 2*np.log(r/r0))


        sol = FEMSolution()
        res = sol.eval("phi", data_number=1)
        for w in res:
            r = np.sqrt(w.x[:,0]**2+w.x[:,1]**2)
            self.assert_allclose(w.data, solution(r), atol=1e-3, rtol=0)

    def amr_2d(self, lib):
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
        model.domainConditions.append(general.Source(rho, geometries=[1]))
        model.boundaryConditions.append(general.DirichletBoundary([True], geometries=[1]))
        model.initialConditions.append(general.InitialCondition(0, geometries=[2]))
        model.initialConditions.append(general.InitialCondition(1, geometries=[1]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        def solution(r):
            return rho/4*np.where(r<=1, r**2-1-2*np.log(r0), 2*np.log(r/r0))


        sol = FEMSolution()
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

        # model: boundary and initial conditions
        model = general.PoissonModel()
        model.domainConditions.append(general.Source(1, geometries=[1,2,3,4,5,6,7,8]))
        model.boundaryConditions.append(general.DirichletBoundary([True], geometries=[18,23,26,29,31,35,37,39]))
        model.initialConditions.append(general.InitialCondition(0, geometries="all"))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        def solution(r, a, r0, rho):
            return np.where(r<=a, rho*(-r**2/6+a**2/2-a**3/(3*r0)), rho*(-a**3/(3*r0)+a**3/(3*r)))

        sol = FEMSolution()
        res = sol.eval("phi", data_number=1)
        for w in res:
            r = np.sqrt(w.x[:,0]**2+w.x[:,1]**2+w.x[:,2]**2)+1e-16
            self.assert_allclose(w.data, -solution(r, 1, 2, 1), atol=0.1, rtol=0)

    def infinite_3d(self, lib):
        r2 = 2
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Sphere(0, 0, 0, 0.8))
        p.geometries.add(geometry.Box(-1, -1, -1, 2, 2, 2))
        p.geometries.add(geometry.InfiniteVolume(1, 1, 1, r2, r2, r2))
        p.mesher.setRefinement(1)

        # model: boundary and initial conditions
        model = general.PoissonModel()
        model.initialConditions.append(general.InitialCondition(0, geometries="all"))
        model.domainConditions.append(general.Source(1, geometries=[1,2,9,10,11,12,13,14]))
        model.boundaryConditions.append(general.DirichletBoundary([True], geometries=[31,36,39,42,43,44]))
        p.models.append(model)

        # solver
        stationary = StationarySolver(solver="cg", prec="jacobi")
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        def solution(r, a, rho):
            return rho*np.where(r<=a, -r**2/6+a**2/2, a**3/(3*r))


        sol = FEMSolution()
        res = sol.eval("phi", data_number=1)
        for w in [res[0]]:
            r = np.sqrt(w.x[:,0]**2+w.x[:,1]**2+w.x[:,2]**2)+1e-16
            self.assert_allclose(w.data, -solution(r, 0.8, 1), atol=0.02, rtol=0)
