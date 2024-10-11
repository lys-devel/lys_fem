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
        p = FEMProject(1)

        # geometry
        p.geometries.scale=1e-6
        p.geometries.add(geometry.Line(0, 0, 0, 1e-6, 0, 0))
        p.geometries.add(geometry.Line(1e-6, 0, 0, 2e-6, 0, 0))
        p.mesher.setRefinement(4)

        # material
        param = llg.LLGParameters(alpha=5, Ms=1e6, Ku=1e3, Aex=1e-11, u_Ku=[0,0,1])
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        x,y,z = sp.symbols("x,y,z")
        model = llg.LLGModel()

        mz = -(x-1e-6)/1e-6
        my = sp.sqrt(1 - mz**2)
        model.initialConditions.append(llg.InitialCondition([0, my, mz], geometries="all"))
        model.domainConditions.append(llg.UniaxialAnisotropy(geometries="all"))
        model.domainConditions.append(llg.GilbertDamping(geometries="all"))
        model.boundaryConditions.append(llg.DirichletBoundary([True, True, True], geometries=[1,3]))
        p.models.append(model)

        # solver
        solver = RelaxationSolver(dt0=1e-9)
        
        p.solvers.append(solver)

        # solve
        lib.run(p)

        def solution(x, A, K):
            return 2*np.arctan(np.exp(np.sqrt(K/A)*x))-np.pi/2

        # solution
        sol = FEMSolution()
        m = sol.eval("m[2]", data_number=0, coords=[0, 1e-6, 2e-6])
        self.assert_array_almost_equal(m, [1,0,-1])

        for i in range(0,10):
            m = sol.eval("m[0]**2+m[1]**2+m[2]**2", data_number=i)
            self.assert_array_almost_equal(m, 1, decimal=2)

        res = sol.eval("m[2]", data_number=-1)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.sin(solution(w.x[:,0]-1e-6, 1e-11, 1e3)), decimal=2)

    def domainWall_3d(self, lib):
        p = FEMProject(2)

        # geometry
        s = 1e-6
        p.geometries.scale=s
        p.geometries.add(geometry.Rect(0, 0, 0, s, s/10))
        p.geometries.add(geometry.Rect(s, 0, 0, s, s/10))
        p.mesher.setRefinement(3)

        # material
        param = llg.LLGParameters(alpha=5, Ms=1e6, Ku=1e3, Aex=1e-11, u_Ku=[0,0,1])
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        x,y,z = sp.symbols("x,y,z")
        model = llg.LLGModel()

        mz = -(x-s)/s
        my = sp.sqrt(1 - mz**2)
        model.initialConditions.append(llg.InitialCondition([0, my, mz], geometries="all"))
        model.domainConditions.append(llg.UniaxialAnisotropy(geometries="all"))
        model.domainConditions.append(llg.GilbertDamping(geometries="all"))
        model.boundaryConditions.append(llg.DirichletBoundary([True, True, True], geometries=[4,6]))
        p.models.append(model)

        # solver
        #solver = StationarySolver()
        solver = RelaxationSolver(dt0=1e-18, dx=0.05, factor=5)
        #solver = TimeDependentSolver(1e-10, 1e-7)
        
        p.solvers.append(solver)


        try:
            # solve
            lib.run(p)
        except:
            pass

        def solution(x, A, K):
            return 2*np.arctan(np.exp(np.sqrt(K/A)*x))-np.pi/2

        # solution
        sol = FEMSolution()
        for i in range(1,10):
            m = sol.eval("m[0]**2+m[1]**2+m[2]**2", data_number=i)
            #self.assert_array_almost_equal(m, 1, decimal=2)

        res = sol.eval("m[2]", data_number=-1)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.sin(solution(w.x[:,0]-1e-6, 1e-11, 1e3)), decimal=2)


    def anisU(self, lib):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1e-6, 0, 0))

        # material
        param = llg.LLGParameters(alpha=1, Ms=1e5, Ku=1e5, u_Ku=[1,0,1])
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

    def deformation(self, lib):
        return
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
            return np.where(r<=a, Ms*(x*1.01+y*0.99+z)/3/np.sqrt(3), np.nan)
        
        def solution2(x, y, z, a, Ms):
            r = np.sqrt(x**2+y**2+z**2)-1e-16
            return np.where(r<=a, Ms*(x*0.32934*1.01+y*0.33734*0.99+z*0.33330)/np.sqrt(3), np.nan)


        sol = FEMSolution()
        res = sol.eval("phi", data_number=1)
        for w in [res[0]]:
            self.assert_allclose(w.data, -solution2(w.x[:,0],w.x[:,1],w.x[:,2], 0.8, 1), atol=0.002, rtol=0)
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
        p = FEMProject(1)

        # geometry
        p.geometries.scale=1e-9
        p.geometries.add(geometry.Line(0, 0, 0, 1e-9, 0, 0))
        p.mesher.setRefinement(0)

        # material
        param = llg.LLGParameters(alpha=0, Aex=0)
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(llg.InitialCondition([1, 0, 0], geometries="all"))
        model.domainConditions.append(llg.ExternalMagneticField([0,0,1], geometries="all"))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(T/100/factor, T/2, steps=[SolverStep(solver="sparsecholesky")])
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


    def damping(self, lib):
        p = FEMProject(1)

        # geometry
        p.geometries.scale=1e-9
        p.geometries.add(geometry.Line(0, 0, 0, 1e-9, 0, 0))
        p.mesher.setRefinement(0)

        # material
        param = llg.LLGParameters(alpha=1, Aex=0)
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(llg.InitialCondition([1, 0, 0], geometries="all"))
        model.domainConditions.append(llg.ExternalMagneticField([0,0,1], geometries="all"))
        model.domainConditions.append(llg.GilbertDamping(geometries="all"))
        p.models.append(model)

        # solver
        solver = RelaxationSolver(dt0=1e-13)
        p.solvers.append(solver)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution()
        res = sol.eval("m[2]", data_number=-2)
        for w in res:
            self.assert_array_almost_equal(w.data, np.ones(w.data.shape), decimal=4)

    def scalar(self, lib):
        factor = 1
        z = sp.Symbol("z")
        p = FEMProject(3)

        # geometry
        p.geometries.scale=1e-9
        p.geometries.add(geometry.Box(0, 0, 0, 1e-9, 0.1e-9, 0.1e-9))
        p.mesher.setRefinement(0)

        # material
        param = llg.LLGParameters(alpha=0, Aex=0)
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model1 = llg.LLGModel()
        model1.initialConditions.append(llg.InitialCondition([1, 0, 0], geometries="all"))
        model1.domainConditions.append(llg.MagneticScalarPotential(z/1.25663706e-6, geometries="all"))
        p.models.append(model1)

        # solver
        solver = TimeDependentSolver(T/100/factor, T/2, steps=[SolverStep(["m", "m_lam"])])
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
            self.assert_array_almost_equal(w.data, -np.ones(w.data.shape), decimal=3)
        res = sol.eval("m[0]", data_number=50*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.ones(w.data.shape), decimal=3)
        res = sol.eval("m[1]", data_number=50*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)

    def scalar_em(self, lib):
        factor = 1
        z = sp.Symbol("z")
        p = FEMProject(3)

        # geometry
        p.geometries.scale=1e-9
        p.geometries.add(geometry.Box(0, 0, 0, 1e-9, 0.1e-9, 0.1e-9))
        p.mesher.setRefinement(0)

        # material
        param = llg.LLGParameters(alpha=0, Aex=0)
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model1 = llg.LLGModel()
        model1.initialConditions.append(llg.InitialCondition([1, 0, 0], geometries="all"))
        model1.domainConditions.append(llg.MagneticScalarPotential("phi", geometries="all"))
        p.models.append(model1)

        model2 = em.MagnetostaticsModel()
        model2.initialConditions.append(em.InitialCondition(z/1.25663706e-6, geometries="all"))
        p.models.append(model2)

        # solver
        solver = TimeDependentSolver(T/100/factor, T/2, steps=[SolverStep(["m", "m_lam"])])
        p.solvers.append(solver)

        # solve
        lib.run(p)

        # solution
        sol = FEMSolution()
        res = sol.eval("phi", data_number=0)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:,2]/1.25663706e-6, decimal=2)
        res = sol.eval("phi", data_number=-1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:,2]/1.25663706e-6, decimal=2)
        res = sol.eval("m[0]", data_number=25*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)
        res = sol.eval("m[1]", data_number=25*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.ones(w.data.shape), decimal=3)
        res = sol.eval("m[0]", data_number=50*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.ones(w.data.shape), decimal=3)
        res = sol.eval("m[1]", data_number=50*factor)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)


    def demagnetization_em(self, lib):
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
        param = llg.LLGParameters(alpha=0, Aex=0, Ms=1)
        p.materials.append(Material([param], geometries=domain))

        # poisson equation for infinite boundary
        model1 = em.MagnetostaticsModel()
        model1.initialConditions.append(em.InitialCondition(0, geometries="all"))
        model1.boundaryConditions.append(em.DirichletBoundary(True, geometries=infBdr))
        model1.domainConditions.append(em.DivSource("Ms*m", geometries=domain))
        p.models.append(model1)

        # model: boundary and initial conditions
        model2 = llg.LLGModel([llg.LLGEquation(geometries=domain)])
        model2.initialConditions.append(llg.InitialCondition([0, 0, 1], geometries=domain))
        p.models.append(model2)

        # solver
        stationary = StationarySolver(steps=[SolverStep(["phi"])])
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