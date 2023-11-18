import numpy as np

from lys_fem import geometry, mf
from lys_fem.fem import FEMProject, DirichletBoundary, InitialCondition, CGSolver, TimeDependentSolver, StationarySolver, BackwardEulerSolver, GMRESSolver, FEMSolution
from lys_fem.models import llg

from .base import FEMTestCase

T = 2*np.pi/1.760859770e11

class testLLG_test(FEMTestCase):
    def test_precession(self):
        return
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 1, 1))

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(InitialCondition("Initial condition1", [1, 0, 0], [1]))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver([model], [BackwardEulerSolver(GMRESSolver())], T/100, T/2)
        p.solvers.append(solver)

        # solve
        mf.run(p)

        # solution
        sol = FEMSolution(".", p)
        res = sol.eval("mx", data_number=25)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)
        res = sol.eval("my", data_number=25)
        for w in res:
            self.assert_array_almost_equal(w.data, np.ones(w.data.shape), decimal=3)
        res = sol.eval("mx", data_number=50)
        for w in res:
            self.assert_array_almost_equal(w.data, -np.ones(w.data.shape), decimal=3)
        res = sol.eval("my", data_number=50)
        for w in res:
            self.assert_array_almost_equal(w.data, np.zeros(w.data.shape), decimal=2)

    def test_singleStep(self):
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 1, 1))

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(InitialCondition("Initial condition1", [1, 0, 0], [1]))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver([model], [BackwardEulerSolver(GMRESSolver())], T/100, T/100)
        p.solvers.append(solver)

        m = mf.run(p, run=False)[0][0]

        # check residual
        m.update(m.x0)
        dt = 0.01
        M, K, b, x = m.M, m.K, m.b, m.x0
        R = mf.mfem.Vector(x.Size())
        M.Mult((x-m.x0)*(1/dt), R)
        K.AddMult(x, R)
        R -=b
        return
        Rp = m.dualToPrime(R)
        val = m.getNodalValue(Rp, 0)
        self.assert_array_almost_equal(val, [0, -1, 0, 0])
        return
        
        # check Jacobian
        Jx = mf.mfem.BlockVector(m.x0)
        gM, gK = m.grad_Mx, m.grad_Kx
        self._gmt =gM*(1/dt)
        self._op= self._gmt + gK
        self._op.Mult(m.x0, Jx)
        Jxp = m.dualToPrime(Jx)
        val = m.getNodalValue(Jxp, 0)
        self.assert_array_almost_equal(val, [1/dt, -1, 0, 2], decimal=4)

    def test_precession(self):
        return 
        p = FEMProject(3)

        # geometry
        p.geometries.add(geometry.Box(0, 0, 0, 1, 1, 1))

        # model: boundary and initial conditions
        model = llg.LLGModel()
        model.initialConditions.append(InitialCondition("Initial condition1", [1, 0, 0], [1]))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver([model], [BackwardEulerSolver(GMRESSolver())], T/100, T/50)
        p.solvers.append(solver)

        # solve
        mf.run(p)

        # solution