import numpy as np

from lys_fem import geometry, mf
from lys_fem.fem import FEMProject, DirichletBoundary, InitialCondition, CGSolver, TimeDependentSolver, StationarySolver, BackwardEulerSolver, GMRESSolver, FEMSolution
from lys_fem.models import llg

from .base import FEMTestCase

T = 2*np.pi/1.760859770e11

class testLLG_test(FEMTestCase):
    def test_precession(self):
        factor = 1
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 0.1, 0.1))
        p.mesher.setRefinement(0)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(InitialCondition("Initial condition1", [1, 0, 0], [1]))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver([model], [BackwardEulerSolver(GMRESSolver())], T/100/factor, T/2)
        p.solvers.append(solver)

        # solve
        mf.run(p)

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

    def test_singleStep(self):
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 0.1, 0.1))
        p.mesher.setRefinement(1)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(InitialCondition("Initial condition1", [1, 0, 0], [1]))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver([model], [BackwardEulerSolver(GMRESSolver())], T/100, T/100)
        p.solvers.append(solver)

        geom, mesh, material, models, solvers = mf.run(p, run=False)
        m = models[0]

        dt, B = 0.01, 1
        # single newton step
        R, J, x1 = self.__RJx(m, m.x0, dt)

        # check residual
        Rp = m.dualToPrime(R)
        val = m.getNodalValue(Rp, 0)
        self.assert_array_almost_equal(val, [0, -B, 0, 0], decimal=4)
        
        # check Jacobian
        Jx = mf.mfem.BlockVector(m.x0)
        J.Mult(m.x0, Jx)
        Jxp = m.dualToPrime(Jx)
        val = m.getNodalValue(Jxp, 0)
        self.assert_array_almost_equal(val, [1/dt, -B, 0, 2], decimal=4)

        # check solution
        val = m.getNodalValue(x1, 0)
        sol1 = self.__solution(dt, B=B)
        self.assert_array_almost_equal(val[0:2], sol1[0:2], decimal=4)

        for i in range(10):
            _, _, x1 = self.__RJx(m, x1, dt)
        val = m.getNodalValue(x1, 0)
        self.assert_array_almost_equal(val[0:2], self.__solution(dt, B=B, newton=10)[0:2], decimal=3)
        #print(val[0:2], self.__solution(dt, B=B, newton=10)[0:2])

    def __RJx(self, m, x, dt):
        m.update(m.x0)
        M, K, b = m.M, m.K, m.b
        R = mf.mfem.Vector(x.Size())
        M.Mult((x-m.x0)*(1/dt), R)
        K.AddMult(x, R)
        R -=b

        gM, gK = m.grad_Mx, m.grad_Kx
        self._gmt =gM*(1/dt)
        J= self._gmt + gK

        dx = mf.mfem.BlockVector(m.x0)
        solver, _ = mf.mfem.getSolver("GMRES")
        solver.SetOperator(J)
        solver.Mult(R, dx)
        return R, J, x - dx

    def __solution(self, dt, newton=1, time=1, B=1):
        dti = 1/dt
        kLi = 1e-5
        lag = 1
        x0 = np.array([1,0,0])
        for ti in range(time):
            x = np.array(x0)
            for i in range(newton):
                A = np.array([[dti, B, 2*x[0]*lag], [-B, dti, 2*x[1]*lag], [x[0]*lag, x[1]*lag, kLi]])
                b = np.array([dti*x0[0], dti*x0[1], 1])
                J = np.array([[dti-2*x[2], B, 2*x[0]*lag], [-B, dti-2*x[2], 2*x[1]*lag], [2*x[0]*lag, 2*x[1]*lag, kLi]])
                Axb = (A.dot(x) - b)
                x = x - np.linalg.inv(J).dot(Axb)
            x0 = x
        return x0

    def test_precession3(self):
        return
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 0.1, 0.1))
        p.mesher.setRefinement(2)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(InitialCondition("Initial condition1", [1, 0, 0], [1]))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver([model], [BackwardEulerSolver(GMRESSolver())], T/1000, T/20)
        p.solvers.append(solver)

        # solve
        mf.run(p)

