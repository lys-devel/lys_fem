import sympy as sp
from lys_fem.fem import FEMProject, solver

from ..base import FEMTestCase

class project_test(FEMTestCase):
    def test_scaling(self):
        p = FEMProject(1)
        p.scaling.set(length=1e-3, time=1e-1)
        self.assertAlmostEqual(p.scaling.getScaling("m/s^2"), 1e-1)

        d = p.saveAsDictionary()
        p.reset()
        self.assertAlmostEqual(p.scaling.getScaling("m/s^2"), 1)

        p.loadFromDictionary(d)
        self.assertAlmostEqual(p.scaling.getScaling("m/s^2"), 1e-1)

    def test_parameters(self):
        a,b = sp.symbols("a,b")
        p = FEMProject(1)
        p.parameters[a] = 2
        p.parameters[b] = a**2
        self.assertEqual(p.parameters[b], a**2)

        d = p.saveAsDictionary()
        p.reset()
        self.assertEqual(len(p.parameters), 0)

        p.loadFromDictionary(d)
        self.assertEqual(p.parameters[b], a**2)

    def test_solvers(self):
        p = FEMProject(1)
        p.scaling.set(time=0.2)

        s1 = solver.TimeDependentSolver(step = 0.1, stop=10)
        s2 = solver.TimeDependentSolver(step = 0.2, stop=10)
        p.solvers.append(s1)
        p.solvers.append(s2)
        self.assert_array_almost_equal(p.solvers[0].getStepList(), [0.5]*101)
        self.assert_array_almost_equal(p.solvers[1].getStepList(), [1]*51)

        d = p.saveAsDictionary()
        p.reset()
        self.assertEqual(len(p.solvers), 0)

        p.loadFromDictionary(d)
        self.assert_array_almost_equal(p.solvers[0].getStepList(), [0.5]*101)
        self.assert_array_almost_equal(p.solvers[1].getStepList(), [1]*51)
