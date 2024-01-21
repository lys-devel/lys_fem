import numpy as np
import sympy as sp

from lys_fem import geometry
from lys_fem.fem import FEMProject, TimeDependentSolver, StationarySolver, RelaxationSolver, FEMSolution, Material
from lys_fem.models import llg, general

from ..base import FEMTestCase

g = 1.760859770e11
T = 2*np.pi/g

class LLG_test(FEMTestCase):
    def domainWall(self, lib):
        p = FEMProject(1)
        p.scaling.set(length=1e-7, time=1e-9, mass=1e-21, current=1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1e-6, 0, 0))
        p.geometries.add(geometry.Line(1e-6, 0, 0, 2e-6, 0, 0))
        p.mesher.setRefinement(4)

        # material
        param = llg.LLGParameters(alpha=5, Ms=1e6, Ku=1e3, Aex=1e-11)
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        x,y,z = sp.symbols("x,y,z")
        model = llg.LLGModel()

        mz = -(x-1e-6)/1e-6
        my = 1 - mz**2
        model.initialConditions.append(llg.InitialCondition([0, my, mz], geometries="all"))
        model.domainConditions.append(llg.UniaxialAnisotropy(geometries="all"))
        model.domainConditions.append(llg.GilbertDamping(geometries="all"))
        model.boundaryConditions.append(llg.DirichletBoundary([True, True, True], geometries=[1,3]))
        p.models.append(model)

        n = 9
        # solver
        solver = RelaxationSolver(dt0=1e-9)
        
        p.solvers.append(solver)

        # solve
        lib.run(p)

        def solution(x, A, K):
            return 2*np.arctan(np.exp(np.sqrt(K/A)*x))-np.pi/2

        # solution
        sol = FEMSolution(".", p)
        for i in range(1,n+1):
            m1 = sol.eval("mx", data_number=i)
            m2 = sol.eval("my", data_number=i)
            m3 = sol.eval("mz", data_number=i)
            self.assert_array_almost_equal(m1[0].data**2 + m2[0].data**2+m3[0].data**2, 1, decimal=4)

        res = sol.eval("mz", data_number=n)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.sin(solution(w.x[:,0]-1e-6, 1e-11, 1e3)), decimal=4)

    def anisU(self, lib):
        p = FEMProject(1)
        p.scaling.set(length=1e-7, time=1e-9, mass=1e-21, current=1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1e-6, 0, 0))


        # material
        param = llg.LLGParameters(alpha=1, Ms=1, Ku=1, u_Ku=[1,0,1])
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(llg.InitialCondition([0, 0, -1], geometries="all"))
        model.domainConditions.append(llg.UniaxialAnisotropy(geometries="all"))
        model.domainConditions.append(llg.GilbertDamping(geometries="all"))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(T/10, T*10)
        p.solvers.append(solver)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("mx", data_number=50)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.ones(w.data.shape)/np.sqrt(2), decimal=2)
        res = sol.eval("mz", data_number=50)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.ones(w.data.shape)/np.sqrt(2), decimal=2)

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

        param = llg.LLGParameters(Ms=1)
        mat1 = Material([param], geometries=domain)
        p.materials.append(mat1)

        # poisson equation for infinite boundary
        model = general.PoissonModel()
        model.initialConditions.append(general.InitialCondition(0, geometries="all"))
        model.boundaryConditions.append(general.DirichletBoundary([True], geometries=infBdr))
        p.models.append(model)

        # llg
        model2 = llg.LLGModel(equations=[llg.LLGEquation(geometries=domain)])
        model2.initialConditions.append(llg.InitialCondition([0, 0, 1], geometries=domain))
        model2.domainConditions.append(llg.ExternalMagneticField([0,0,1], geometries=domain))
        model2.domainConditions.append(llg.Demagnetization(geometries=mBdr))
        p.models.append(model2)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        def solution(x, y, z, a, Ms):
            r = np.sqrt(x**2+y**2+z**2)-1e-16
            return np.where(r<=a, Ms/3*z, np.nan)


        sol = FEMSolution(".", p)
        res = sol.eval("phi", data_number=1)
        for w in [res[0]]:
            self.assert_allclose(w.data, -solution(w.x[:,0],w.x[:,1],w.x[:,2], 0.8, 1), atol=0.02, rtol=0)

    def stationary(self, lib):
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 0.1, 0.1))
        p.mesher.setRefinement(0)

        # material
        param = llg.LLGParameters(0)
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(llg.InitialCondition([1/np.sqrt(2), 0, 1/np.sqrt(2)], geometries="all"))
        model.domainConditions.append(llg.ExternalMagneticField([0,0,1], geometries="all"))
        p.models.append(model)

        # solver
        solver = StationarySolver()
        p.solvers.append(solver)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("mx", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)
        res = sol.eval("mz", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, np.ones(w.data.shape), decimal=2)

    def precession(self, lib):
        factor = 1
        p = FEMProject(3)
        p.scaling.set(length=1e-9, time=1e-12, mass=1e-27, current=1e-3)


        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1e-9, 0.1e-9, 0.1e-9))
        p.mesher.setRefinement(0)

        # material
        param = llg.LLGParameters(alpha=0, Aex=0)
        mat1 = Material([param], geometries=[1, 2])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(llg.InitialCondition([1, 0, 0], geometries="all"))
        model.domainConditions.append(llg.ExternalMagneticField([0,0,1], geometries="all"))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(T/100/factor, T/2)
        p.solvers.append(solver)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("mx", data_number=25*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)
        res = sol.eval("my", data_number=25*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, np.ones(w.data.shape), decimal=3)
        res = sol.eval("mx", data_number=50*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.ones(w.data.shape), decimal=3)
        res = sol.eval("my", data_number=50*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)
