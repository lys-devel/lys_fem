import numpy as np
from lys_fem.fem import FEMProject, solver

from ..base import FEMTestCase

class solvers_test(FEMTestCase):
    def test_stationary(self):
        s = solver.StationarySolver()
        self.assertEqual(s.className, "Stationary Solver")

        d = s.saveAsDictionary()
        s = solver.FEMSolver.loadFromDictionary(d)
        self.assertTrue(isinstance(s, solver.StationarySolver))

    def test_tdep(self):
        p = FEMProject(1)
        p.scaling.set(time=0.2)

        s = solver.TimeDependentSolver(step = 0.1, stop=10)
        p.solvers.append(s)
        self.assert_array_almost_equal(s.getStepList(), [0.5]*101)

        d = s.saveAsDictionary()
        s = solver.FEMSolver.loadFromDictionary(d)
        p.solvers.append(s)
        self.assert_array_almost_equal(s.getStepList(), [0.5]*101)

    def test_relax(self):
        p = FEMProject(1)
        p.scaling.set(time=1e-8)

        s = solver.RelaxationSolver(dt0=1e-9, dx=1e-1)
        p.solvers.append(s)

        self.assertAlmostEqual(s.dt0, 0.1)
        self.assertAlmostEqual(s.dx, 0.1)

        d = s.saveAsDictionary()
        s = solver.FEMSolver.loadFromDictionary(d)
        p.solvers.append(s)

        self.assertAlmostEqual(s.dt0, 0.1)
        self.assertAlmostEqual(s.dx, 0.1)
