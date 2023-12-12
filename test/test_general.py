from scipy import special
import numpy as np
import sympy as sp
from numpy.testing import assert_array_almost_equal, assert_allclose

from lys_fem import geometry, mf
from lys_fem.fem import FEMProject, Material, Source, DirichletBoundary, NeumannBoundary, InitialCondition, FEMSolution
from lys_fem.fem import StationarySolver, CGSolver, TimeDependentSolver
from lys_fem.models import general
from lys_fem.mf.weakform import x,y,z

from .base import FEMTestCase

class poisson_test(FEMTestCase):
    def test_3d_dirichlet(self):
        return
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Sphere(0, 0, 0, 2))
        p.geometries.add(geometry.Sphere(0, 0, 0, 5))
        p.mesher.setRefinement(1)

        # model: boundary and initial conditions
        model = general.PoissonModel()
        model.domainConditions.append(Source("Source1", 3, [1]))
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [2]))
        model.initialConditions.append(InitialCondition("Initial condition1", 0, [1,2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver(CGSolver(), [model])
        p.solvers.append(stationary)

        # solve
        mf.run(p)

        def solution(r, a, r0, rho):
            return np.where(r<=a, rho*(-r**2/6+a**2/2-a**3/(3*r0)), rho*(-a**3/(3*r0)+a**3/(3*r)))
                

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("phi", data_number=1)
        for w in res:
            r = np.sqrt(w.x[:,0]**2+w.x[:,1]**2+w.x[:,2]**2)
            assert_array_almost_equal(w.data, solution(r, 2, 5, 3))

    def test_3d(self):
        return
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Sphere(0, 0, 0, 2))
        p.geometries.add(geometry.Sphere(0, 0, 0, 5))
        p.mesher.setRefinement(0)

        # model: boundary and initial conditions
        model = general.PoissonModel()
        model.domainConditions.append(Source("Source1", 3, [1]))
        model.boundaryConditions.append(DirichletBoundary("Dirichlet boundary1", [True], [2]))
        model.initialConditions.append(InitialCondition("Initial condition1", 0, [1,2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver(CGSolver(), [model])
        p.solvers.append(stationary)

        # solve
        mf.run(p)
        return

        def solution(r, a, r0, rho):
            return np.where(r<=a, rho*(-r**2/6+a**2/2-a**3/(3*r0)), rho*(-a**3/(3*r0)+a**3/(3*r)))
                
        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("phi1", data_number=1)
        for w in res:
            print(w.data)
            r = np.sqrt(w.x[:,0]**2+w.x[:,1]**2+w.x[:,2]**2)
            #assert_array_almost_equal(w.data, solution(r, 2, 5, 3))