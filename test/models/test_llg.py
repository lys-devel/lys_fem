import numpy as np
import sympy as sp

from lys_fem import geometry
from lys_fem.fem import FEMProject, TimeDependentSolver, StationarySolver, RelaxationSolver, FEMSolution, Material, SolverStep
from lys_fem.models import llg, em, elasticity

from ..base import FEMTestCase

g = 1.760859770e11
T = 2*np.pi/g

class LLG_test(FEMTestCase):
    def test_domainWall(self, discretization="BackwardEuler"):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1e-6, 0, 0))
        p.geometries.add(geometry.Line(1e-6, 0, 0, 2e-6, 0, 0))
        p.mesher.setRefinement(4)

        # material
        param = llg.LLGParameters(alpha_LLG=10, Ms=1e6, Ku=1e3, Aex=1e-11, u_Ku=[0,0,1])
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        x,y,z = sp.symbols("x,y,z")
        model = llg.LLGModel(discretization=discretization)

        mz = -(x-1e-6)/1e-6
        my = sp.sqrt(1 - mz**2)
        model.initialConditions.append(llg.InitialCondition([0, my, mz], geometries="all"))
        model.domainConditions.append(llg.UniaxialAnisotropy(geometries="all"))
        model.boundaryConditions.append(llg.DirichletBoundary([True, True, True], geometries=[1,3]))
        p.models.append(model)

        # solver
        solver = RelaxationSolver(dt0=1e-12, dt_max=1e-8, dx=0.2, diff_expr="m", solver="pardiso")        
        p.solvers.append(solver)

        # solve
        p.run()

        def solution(x, A, K):
            return 2*np.arctan(np.exp(np.sqrt(K/A)*x))-np.pi/2

        # solution
        sol = FEMSolution()
        m = sol.eval("m[2]", data_number=0, coords=[0, 1e-6, 2e-6])
        self.assert_array_almost_equal(m, [1,0,-1])

        m = sol.eval("norm(m)", data_number=-1)
        self.assert_array_almost_equal(m, 1, decimal=3)

        res = sol.eval("m[2]", data_number=-1)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.sin(solution(w.x[:,0]-1e-6, 1e-11, 1e3)), decimal=2)

    def test_anisU(self, discretization="BackwardEuler"):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1e-6, 0, 0))

        # material
        param = llg.LLGParameters(alpha_LLG=1, Ms=1e5, Ku=1e5, u_Ku=[1,0,1])
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = llg.LLGModel(discretization=discretization)
        model.initialConditions.append(llg.InitialCondition([0, 0, -1], geometries="all"))
        model.domainConditions.append(llg.UniaxialAnisotropy(geometries="all"))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(T/10, T*10, steps=[SolverStep(solver="pardiso", vars=["m", "m_lam"])])
        p.solvers.append(solver)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("m[0]", data_number=50)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.ones(w.data.shape)/np.sqrt(2), decimal=2)
        res = sol.eval("m[2]", data_number=50)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.ones(w.data.shape)/np.sqrt(2), decimal=2)

    def test_anisC(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1e-6, 0, 0))

        # material
        param = llg.LLGParameters(alpha_LLG=1, Ms=1e5, Kc=1e5)
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(llg.InitialCondition([0.8, 0, 0.6], geometries="all"))
        model.domainConditions.append(llg.CubicAnisotropy(geometries="all"))
        p.models.append(model)

        # solver
        solver = RelaxationSolver(dt0=1e-14, dt_max=1e-12, dx=0.1, diff_expr="m", maxiter=3000, tolerance=1e-6, solver="pardiso")
        p.solvers.append(solver)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        B = sol.eval("-4/Ms*einsum('ijkl,j,k,l->i', Kc, m, m, m)", data_number=0, coords=[0])
        B_res = -2*np.array([0.8*0.6**2, 0, 0.6*0.8**2])
        self.assert_array_almost_equal(B, B_res, decimal=2)

        res = sol.eval("m", data_number=-1, coords=[0])
        self.assert_array_almost_equal(res, [1,0,0], decimal=5)

    def test_precession(self, discretization="BackwardEuler"):
        factor = 10
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1e-9, 0, 0))
        p.mesher.setRefinement(0)

        # material
        param = llg.LLGParameters(alpha_LLG=0, Aex=0)
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = llg.LLGModel(discretization=discretization, type="L2")
        model.initialConditions.append(llg.InitialCondition([1, 0, 0], geometries="all"))
        model.domainConditions.append(llg.ExternalMagneticField([0,0,1], geometries="all"))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(T/100/factor, T/2, steps=[SolverStep(solver="pardiso")])
        p.solvers.append(solver)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("m[0]", data_number=0)
        for w in res:
            self.assert_array_almost_equal(w.data, np.ones(w.data.shape), decimal=2)
        res = sol.eval("m[1]", data_number=0)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=3)
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

    def test_damping(self):
        p = FEMProject(1)

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1e-9, 0, 0))
        p.mesher.setRefinement(0)

        # material
        param = llg.LLGParameters(alpha_LLG=1, Aex=0)
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = llg.LLGModel(discretization="BackwardEuler")
        model.initialConditions.append(llg.InitialCondition([1, 0, 0], geometries="all"))
        model.domainConditions.append(llg.ExternalMagneticField([0,0,1], geometries="all"))
        p.models.append(model)

        # solver
        solver = RelaxationSolver(dt0=1e-13)
        p.solvers.append(solver)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("m[2]", data_number=-2)
        for w in res:
            self.assert_array_almost_equal(w.data, np.ones(w.data.shape), decimal=4)

    def test_scalar(self):
        factor = 1
        z = sp.Symbol("z")
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1e-9, 0.1e-9, 0.1e-9))
        p.mesher.setRefinement(0)

        # material
        param = llg.LLGParameters(alpha_LLG=0, Aex=0)
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model1 = llg.LLGModel(discretization="BackwardEuler", type="H1", order=1)
        model1.initialConditions.append(llg.InitialCondition([1, 0, 0], geometries="all"))
        model1.domainConditions.append(llg.MagneticScalarPotential(z/1.25663706e-6, geometries="all"))
        p.models.append(model1)

        # solver
        solver = TimeDependentSolver(T/100/factor, T/2, steps=[SolverStep(["m", "m_lam"])])
        p.solvers.append(solver)

        # solve
        p.run()

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

    def test_scalar_em(self):
        factor = 1
        z = sp.Symbol("z")
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1e-9, 0.1e-9, 0.1e-9))
        p.mesher.setRefinement(0)

        # material
        param = llg.LLGParameters(alpha_LLG=0, Aex=0)
        mat1 = Material([param], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model1 = llg.LLGModel(discretization="BackwardEuler", type="H1", order=1)
        model1.initialConditions.append(llg.InitialCondition([1, 0, 0], geometries="all"))
        model1.domainConditions.append(llg.MagneticScalarPotential("phi", geometries="all"))
        p.models.append(model1)

        model2 = em.MagnetostaticsModel()
        model2.initialConditions.append(em.InitialCondition("z/1.25663706e-6", geometries="all"))
        p.models.append(model2)

        # solver
        solver = TimeDependentSolver(T/100/factor, T/2, steps=[SolverStep(["m", "m_lam"])])
        p.solvers.append(solver)

        # solve
        p.run()

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

    def test_demagnetization_em(self):
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
        param = llg.LLGParameters(alpha_LLG=0, Aex=0, Ms=1)
        p.materials.append(Material([param], geometries=domain))

        # poisson equation for infinite boundary
        model1 = em.MagnetostaticsModel(J="J")
        model1.initialConditions.append(em.InitialCondition(0, geometries="all"))
        model1.boundaryConditions.append(em.DirichletBoundary(True, geometries=infBdr))
        model1.domainConditions.append(em.DivSource("Ms*m", geometries=domain))
        p.models.append(model1)

        # model: boundary and initial conditions
        model2 = llg.LLGModel(geometries=domain)
        model2.initialConditions.append(llg.InitialCondition([0, 0, 1], geometries=domain))
        p.models.append(model2)

        # solver
        stationary = StationarySolver(steps=[SolverStep(["phi"])])
        p.solvers.append(stationary)

        # solve
        p.run()

        def solution(x, y, z, a, Ms):
            r = np.sqrt(x**2+y**2+z**2)-1e-16
            return np.where(r<=a, Ms/3*z, np.nan)


        sol = FEMSolution()
        res = sol.eval("phi", data_number=1)
        for w in [res[0]]:
            self.assert_allclose(w.data, solution(w.x[:,0],w.x[:,1],w.x[:,2], 0.8, 1), atol=0.02, rtol=0)

    def test_thermal2(self):
        return
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(-5e-9, -5e-9, -5e-9, 10e-9, 10e-9, 10e-9))
        p.mesher.addSizeConstraint(size=0.7e-9, geometries="all")
        p.mesher.addTransfinite(geometries="all")
        p.randomFields.add("R", "L2", shape=(3,), tdep = True)

        # Material
        param = llg.LLGParameters(alpha=0.1, Aex=1.3e-11, Ms=8e5)
        p.materials.append(Material([param], geometries="all"))

        # model: boundary and initial conditions
        model = llg.LLGModel([llg.LLGEquation(geometries="all")], type="H1", order=1)
        model.initialConditions.append(llg.InitialCondition([1, 0, 0], geometries="all"))
        model.domainConditions.append(llg.ThermalFluctuation(T=100, R="R", geometries="all"))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(1e-13, 0.01e-9, solver="pardiso")
        p.solvers.append(solver)

        # solve
        p.run()

        sol = FEMSolution()
        import matplotlib.pyplot as plt
        plt.plot([(sol.integrate("m[0]", data_number=i*10))/1e-24 for i in range(100)])
        plt.show()
        
    def test_magnetoStriction(self):
        p = FEMProject(3)

        C1, C2, C4 = 100e9, 40e9, 20e9
        Ms = 1e5
        l100, l111 = 1e-6, 2e-6
        rat = 1.2
        B1, B2 = 3*l100*(C2-C1)/2, -3*l111*C4
        e11, e22, e33, e12, e23, e31 = 1e-2, rat*1e-2, 1e-2, (rat+1)/2*1e-2, (rat+1)/2*1e-2, 1e-2

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1e-6, 1e-6, 1e-7))
        p.mesher.addSizeConstraint(geometries="all", size=1e-6)

        # material
        param = llg.LLGParameters(alpha_LLG=20, Ms=Ms, B1=B1, B2=B2)
        param2 = elasticity.ElasticParameters(C=[C1, C2, C4], type="cubic")
        param3 = llg.UserDefinedParameters(u=["1e-2*(x+y+z)", str(rat)+"e-2*(x+y+z)", "1e-2*(x+y+z)"])
        mat1 = Material([param, param2, param3], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = llg.LLGModel(order=1)
        model.initialConditions.append(llg.InitialCondition([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], geometries="all"))
        model.domainConditions.append(llg.CubicMagnetoStriction("u", geometries="all"))
        p.models.append(model)

        # solver
        solver = RelaxationSolver(dt0=1e-4, dt_max=1e-9, dx=0.01, diff_expr="m", maxiter=300, tolerance=1e-6, solver="pardiso")
        p.solvers.append(solver)

        # solve
        p.run()

        a,b,c = -B2*e12, -B2*e31-2*B1*e11+2*B1*e22, 2*B2*e12
        r1, r2 = (-b+np.sqrt(b**2-4*a*c))/(2*a),(-b-np.sqrt(b**2-4*a*c))/(2*a)

        # solution
        sol = FEMSolution()
        s = sol.eval("m", data_number=-1)[0].data[0]

        self.assert_allclose(r1, s[1]/s[0], rtol=1e-4)

    def test_magnetoRotation(self):
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1e-6, 1e-6, 1e-7))
        p.mesher.addSizeConstraint(geometries="all", size=1e-6)

        # material
        param = llg.LLGParameters(alpha_LLG=10, Ms=1e6, Kc=1e3, Aex=0)
        param2 = llg.UserDefinedParameters(u=["-z", "0", "x"])
        mat1 = Material([param, param2], geometries="all")
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = llg.LLGModel(order=1)
        model.initialConditions.append(llg.InitialCondition([0, 0, 1], geometries="all"))
        model.domainConditions.append(llg.CubicAnisotropy(geometries="all"))
        model.domainConditions.append(llg.CubicMagnetoRotationCoupling("u*1e-3", geometries="all"))
        p.models.append(model)

        # solver
        solver = RelaxationSolver(dt0=1e-11, dt_max=1e-8, dx=0.01, diff_expr="m", maxiter=300, tolerance=1e-9, solver="pardiso")
        p.solvers.append(solver)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        w = np.array(sol.eval("(grad(u*0.001).T-grad(u*0.001))/2", 0, coords=[(0,0,0)])).reshape((3,3))
        self.assert_array_almost_equal(w, np.array([[0,0,-1], [0,0,0], [1,0,0]])*0.001)

        s = sol.eval("m", data_number=-1)[0].data[0]
        self.assertTrue(s[0]/s[2] < 0)
        self.assertAlmostEqual(abs(s[0]/s[2]), 1e-3, delta=1e-3)
