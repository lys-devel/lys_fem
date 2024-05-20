import numpy as np
import sympy as sp

from lys_fem import geometry
from lys_fem.fem import FEMProject, TimeDependentSolver, StationarySolver, RelaxationSolver, FEMSolution, Material, SolverStep
from lys_fem.models import llg, em, elasticity

from ..base import FEMTestCase

g = 1.760859770e11
T = 2*np.pi/g

class LLG_test(FEMTestCase):
    def domainWall(self, lib):
        Aex = sp.symbols("Aex")

        p = FEMProject(1)
        p.scaling.set(length=1e-9, time=1e-9, mass=1e-21, current=1)
        p.parameters[Aex] = 1e-11

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1e-6, 0, 0))
        p.geometries.add(geometry.Line(1e-6, 0, 0, 2e-6, 0, 0))
        p.mesher.setRefinement(4)

        # material
        param = llg.LLGParameters(alpha=5, Ms=1e6, Ku=1e3, Aex=Aex)
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

        n = 13
        # solver
        solver = RelaxationSolver(dt0=1e-9)
        
        p.solvers.append(solver)

        # solve
        lib.run(p)

        def solution(x, A, K):
            return 2*np.arctan(np.exp(np.sqrt(K/A)*x))-np.pi/2

        # solution
        sol = FEMSolution()
        for i in range(1,n+1):
            m = sol.eval("m[0]**2+m[1]**2+m[2]**2", data_number=i)
            self.assert_array_almost_equal(m, 1, decimal=4)

        res = sol.eval("m[2]", data_number=n)
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
        sol = FEMSolution()
        res = sol.eval("m[0]", data_number=50)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.ones(w.data.shape)/np.sqrt(2), decimal=2)
        res = sol.eval("m[2]", data_number=50)
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
        model = em.MagnetostaticsModel()
        model.initialConditions.append(em.MagnetostaticInitialCondition(0, geometries="all"))
        model.boundaryConditions.append(em.DirichletBoundary([True], geometries=infBdr))
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


        sol = FEMSolution()
        res = sol.eval("phi", data_number=1)
        for w in [res[0]]:
            self.assert_allclose(w.data, -solution(w.x[:,0],w.x[:,1],w.x[:,2], 0.8, 1), atol=0.02, rtol=0)

    def deformation(self, lib):
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Sphere(0, 0, 0, 0.8))
        p.geometries.add(geometry.Box(-1, -1, -1, 2, 2, 2))
        p.geometries.add(geometry.InfiniteVolume(1, 1, 1, 2, 2, 2))
        p.mesher.setRefinement(2)

        domain = [1,2,9,10,11,12,13,14]
        infBdr =[31,36,39,42,43,44]
        mBdr = [1,5,8,11,14,16,24,25]

        param_llg = llg.LLGParameters(Ms=1)
        param_ela = elasticity.ElasticParameters()

        mat1 = Material([param_llg, param_ela], geometries=domain)
        p.materials.append(mat1)

        # poisson equation for infinite boundary
        model = em.MagnetostaticsModel()
        model.initialConditions.append(em.MagnetostaticInitialCondition(0, geometries="all"))
        model.boundaryConditions.append(em.DirichletBoundary([True], geometries=infBdr))
        p.models.append(model)

        # llg
        model2 = llg.LLGModel(equations=[llg.LLGEquation(geometries=domain)])
        model2.initialConditions.append(llg.InitialCondition([1/np.sqrt(3)]*3, geometries=domain))
        model2.domainConditions.append(llg.Demagnetization(geometries=mBdr))
        p.models.append(model2)

        # elasticity
        x,y = sp.symbols("x,y")
        model3 = elasticity.ElasticModel(3, equations=[elasticity.ChristffelEquation(geometries=domain)])
        model3.initialConditions.append(elasticity.InitialCondition([x*0.01, -y*0.01, 0], geometries=domain))
        p.models.append(model3)

        # solver
        step = SolverStep(["phi"], deformation="u")
        stationary = StationarySolver(steps=[step])
        p.solvers.append(stationary)

        # solve
        lib.run(p)

        def solution(x, y, z, a, Ms):
            r = np.sqrt(x**2+y**2+z**2)-1e-16
            return np.where(r<=a, Ms*(x+y+z)/3/np.sqrt(3), np.nan)
        
        def solution2(x, y, z, a, Ms):
            r = np.sqrt(x**2+y**2+z**2)-1e-16
            return np.where(r<=a, Ms*(x*0.32934+y*0.33734+z*0.33330)/np.sqrt(3), np.nan)


        sol = FEMSolution()
        res = sol.eval("phi", data_number=1)
        for w in [res[0]]:
            #self.assert_allclose(-solution2(w.x[:,0],w.x[:,1],w.x[:,2], 0.8, 1), -solution(w.x[:,0],w.x[:,1],w.x[:,2], 0.8, 1), atol=0.002, rtol=0)
            self.assert_allclose(w.data, -solution(w.x[:,0],w.x[:,1],w.x[:,2], 0.8, 1), atol=0.002, rtol=0)

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
        sol = FEMSolution()
        res = sol.eval("m[0]", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)
        res = sol.eval("m[2]", data_number=1)
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
        sol = FEMSolution()
        res = sol.eval("m[0]", data_number=25*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)
        res = sol.eval("m[1]", data_number=25*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, np.ones(w.data.shape), decimal=3)
        res = sol.eval("m[0]", data_number=50*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.ones(w.data.shape), decimal=3)
        res = sol.eval("m[1]", data_number=50*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)
