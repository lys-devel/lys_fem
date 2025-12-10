import numpy as np

from lys_fem import geometry
from lys_fem.fem import FEMProject, StationarySolver, FEMSolution, Material
from lys_fem.models import semiconductor as sc
from lys_fem.models import em

from ..base import FEMTestCase


class semiconductor_test(FEMTestCase):
    def test_stationary(self):
        p = FEMProject(1)
        scale = 20

        ni = 1e16
        Na = 2e20
        Nd = 1e20
        T = 300

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, scale*1e-6, 0, 0))
        p.geometries.add(geometry.Line(scale*1e-6, 0, 0, scale*2e-6, 0, 0))
        p.mesher.setRefinement(8)

        # material
        param_e = em.ElectrostaticParameters(eps_r=np.eye(3)*11.8)
        param1 = sc.SemiconductorParameters(mu_n=0.15, mu_p=0.05, N_a=Na)
        param2 = sc.SemiconductorParameters(mu_n=0.15, mu_p=0.05, N_d=Nd)
        param3 = sc.UserDefinedParameters(T=T)
        p.materials.append(Material([param_e, param1, param3], geometries=[1]))
        p.materials.append(Material([param_e, param2, param3], geometries=[2]))

        # model: boundary and initial conditions
        model = sc.SemiconductorModel()
        model.initialConditions.append(sc.InitialCondition.fromDensities(ni, Na=Na, geometries=[1]))
        model.initialConditions.append(sc.InitialCondition.fromDensities(ni, Nd=Nd, geometries=[2]))
        model.boundaryConditions.append(sc.DirichletBoundary([True, True], geometries=[1,3]))
        p.models.append(model)

        # poisson equation
        model = em.ElectrostaticsModel()
        model.initialConditions.append(em.InitialCondition(0, geometries="all"))
        p.models.append(model)

        # solver
        solver = StationarySolver()
        
        p.solvers.append(solver)

        # solve
        p.run()

        # solution
        kB, q = 1.3806488e-23, 1.602176634e-19
        sol = FEMSolution()
        c = np.linspace(0,scale*2e-6, 300)
        V = sol.eval("phi", data_number=1, coords=c)
        self.assertAlmostEqual(V[-1]-V[0], kB*T/q*np.log(Na*Nd/ni/ni), places=5)
