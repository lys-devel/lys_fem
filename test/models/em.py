
import numpy as np

from lys_fem import geometry
from lys_fem.fem import FEMProject, FEMSolution, StationarySolver
from lys_fem.models import em

from ..base import FEMTestCase

class magnetostatistics_test(FEMTestCase):
    def dirichlet_2d(self, lib):
        p = FEMProject(2)
        p.scaling.set(current=100)

        r0 = 3
        a = 1
        rho = 1
        # geometry
        p.geometries.add(geometry.Disk(0, 0, 0, a, a))
        p.geometries.add(geometry.Disk(0, 0, 0, r0, r0))
        p.mesher.setRefinement(4)

        # model: boundary and initial conditions
        model = em.MagnetostatisticsModel()
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

