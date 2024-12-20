
import numpy as np

from lys_fem import geometry
from lys_fem.fem import FEMProject, FEMSolution, StationarySolver, Material
from lys_fem.models import em

from ..base import FEMTestCase

class magnetostatistics_test(FEMTestCase):
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
        model = em.MagnetostaticsModel()
        model.domainConditions.append(em.Source(rho, geometries=[1]))
        model.boundaryConditions.append(em.DirichletBoundary([True], geometries=[1]))
        model.initialConditions.append(em.InitialCondition(0, geometries=[2]))
        model.initialConditions.append(em.InitialCondition(1, geometries=[1]))
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

    def demagnetization(self, lib):
        r2 = 2
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Sphere(0, 0, 0, 0.8))
        p.geometries.add(geometry.Box(-1, -1, -1, 2, 2, 2))
        p.geometries.add(geometry.InfiniteVolume(1, 1, 1, r2, r2, r2))
        p.mesher.setRefinement(1)

        domain = [1,2,9,10,11,12,13,14]
        infBdr =[31,36,39,42,43,44]
        mBdr = [1,5,8,11,14,16,24,25]

        # Material
        param = em.UserDefinedParameters(M=1)
        p.materials.append(Material([param], geometries=domain))

        # poisson equation for infinite boundary
        model = em.MagnetostaticsModel()
        model.initialConditions.append(em.InitialCondition(0, geometries="all"))
        model.boundaryConditions.append(em.DirichletBoundary(True, geometries=infBdr))
        model.domainConditions.append(em.DivSource(np.array([0,0,"M"]), geometries=domain))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        def solution(x, y, z, a, Ms):
            r = np.sqrt(x**2+y**2+z**2)-1e-16
            return np.where(r<=a, Ms/3*z, np.nan)

        sol = FEMSolution()
        res = sol.eval("phi", data_number=1)
        for w in [res[0]]:
            self.assert_allclose(w.data, solution(w.x[:,0],w.x[:,1],w.x[:,2], 0.8, 1), atol=0.02, rtol=0)
