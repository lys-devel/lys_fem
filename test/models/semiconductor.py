import numpy as np

from lys_fem import geometry
from lys_fem.fem import FEMProject, TimeDependentSolver, StationarySolver, FEMSolution, Material
from lys_fem.models import semiconductor as sc
from lys_fem.models import em

from ..base import FEMTestCase


class semiconductor_test(FEMTestCase):
    def stationary(self, lib):
        p = FEMProject(1)
        p.scaling.set(length=1e-9, time=1e-12, mass=1e-27)
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
        param1 = sc.SemiconductorParameters(eps_r = 11.8, mu_n=0.15, mu_p=0.05, T=T, N_a=Na)
        param2 = sc.SemiconductorParameters(eps_r = 11.8, mu_n=0.15, mu_p=0.05, T=T, N_d=Nd)
        p.materials.append(Material([param1], geometries=[1]))
        p.materials.append(Material([param2], geometries=[2]))

        # model: boundary and initial conditions
        model = sc.SemiconductorModel()
        model.initialConditions.append(sc.InitialCondition.fromDensities(ni, Na=Na, geometries=[1]))
        model.initialConditions.append(sc.InitialCondition.fromDensities(ni, Nd=Nd, geometries=[2]))
        model.boundaryConditions.append(sc.DirichletBoundary([True, True], geometries=[1,3]))
        p.models.append(model)

        # poisson equation
        model = em.ElectrostaticsModel()
        model.initialConditions.append(em.ElectrostaticInitialCondition(0, geometries="all"))
        p.models.append(model)

        # solver
        solver = StationarySolver()
        
        p.solvers.append(solver)

        # solve
        lib.run(p)

        # solution
        kB, q = 1.3806488e-23, 1.602176634e-19
        sol = FEMSolution()
        c = np.linspace(0,scale*2e-6, 300)
        V = sol.eval("phi", data_number=1, coords=c)
        self.assertAlmostEqual(V[-1]-V[0], kB*T/q*np.log(Na*Nd/ni/ni), places=5)
