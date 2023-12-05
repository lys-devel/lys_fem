import numpy as np

from lys_fem import geometry, mf
from lys_fem.fem import FEMProject, DirichletBoundary, InitialCondition, CGSolver, TimeDependentSolver, StationarySolver, BackwardEulerSolver, GMRESSolver, FEMSolution, Material
from lys_fem.models import llg

from .base import FEMTestCase

g = 1.760859770e11
T = 2*np.pi/g

class LLG_test(FEMTestCase):
    def test_stationary(self):
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 0.1, 0.1))
        p.mesher.setRefinement(0)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(InitialCondition("Initial condition1", [1/np.sqrt(2), 0, 1/np.sqrt(2)], [1]))
        p.models.append(model)

        # solver
        solver = StationarySolver([model], [GMRESSolver()])
        p.solvers.append(solver)

        # solve
        mf.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("mx", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)

    def test_precession(self):
        factor = 1
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 0.1, 0.1))
        p.mesher.setRefinement(0)

        # material
        param = llg.LLGParameters(0)
        mat1 = Material("Material1", [1, 2], [param])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(InitialCondition("Initial condition1", [1, 0, 0], [1]))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(BackwardEulerSolver(GMRESSolver()), [model], T/100/factor, T/2)
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

        # material
        param = llg.LLGParameters(0)
        mat1 = Material("Material1", [1], [param])
        p.materials.append(mat1)

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(InitialCondition("Initial condition1", [1, 0, 0], [1]))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(BackwardEulerSolver(GMRESSolver()), [model], T/100, T/100)
        p.solvers.append(solver)

        mesh, material, models, solvers = mf.run(p, run=False)
        m = solvers[0]._model

        # check m0
        val = m.getNodalValue(m.x0)
        self.assert_array_almost_equal(val[:,1], [1, 0, 0, 0], decimal=4)

        dt, B = 0.02/g, 1
        # single newton step
        b, R, J, x1 = self.__RJx(m, m.x0, dt)

        # check rhs b
        bp = m.dualToPrime(b)
        val = m.getNodalValue(bp)
        self.assert_array_almost_equal(val[:,0], [1/dt/g, 0, 0, 1], decimal=4)

        # check residual
        Rp = m.dualToPrime(R)
        val = m.getNodalValue(Rp)
        self.assert_array_almost_equal(val[:,0], [0, -B, 0, 0], decimal=4)
        
        # check Jacobian
        Jx = mf.mfem.BlockVector(m.x0)
        J.Mult(m.x0, Jx)
        Jxp = m.dualToPrime(Jx)
        val = m.getNodalValue(Jxp)
        self.assert_array_almost_equal(val[:,0], [1/dt/g, -B, 0, 2], decimal=4)

        # check solution
        val = m.getNodalValue(x1)
        sol1 = self.__solution(dt, B=B)
        self.assert_array_almost_equal(val[0:2, 0], sol1[0:2], decimal=4)

        for i in range(10):
            _, _, _, x1 = self.__RJx(m, x1, dt)
        val = m.getNodalValue(x1)
        self.assert_array_almost_equal(val[0:2, 0], self.__solution(dt, B=B, newton=10)[0:2], decimal=3)
        print(val[0:2, 0], self.__solution(dt, B=B, newton=10)[0:2])

    def __RJx(self, m, x, dt):
        m.dt = dt
        m._updateCoefficients()
        m.update(m.x0)
        K, b = m.K, m.b
        R = mf.mfem.Vector(x.Size())
        K.Mult(x, R)
        R -=b

        J = m.gK

        dx = mf.mfem.BlockVector(m.x0)
        solver, _ = mf.mfem.getSolver("GMRES")
        solver.SetOperator(J)
        solver.Mult(R, dx)
        return b, R, J, x - dx

    def __solution(self, dt, newton=1, time=1, B=1):
        dti = 1/dt/g
        kLi = 1e-5 * 0
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

