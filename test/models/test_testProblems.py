import os
import numpy as np
import sympy as sp

from numpy.testing import assert_array_almost_equal

from lys_fem import geometry
from lys_fem.fem import FEMProject, StationarySolver, TimeDependentSolver, FEMSolution, SolverStep, Material, mpi
from lys_fem.models import test

from ..base import FEMTestCase

class testProblems_test(FEMTestCase):
    def test_linear(self):
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # model: boundary and initial conditions
        model = test.LinearTestModel()
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(test.InitialCondition(0.0, geometries=[1]))
        model.initialConditions.append(test.InitialCondition(2.0, geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver(steps=[SolverStep()])
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("X", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

    def test_cond(self):
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # model: boundary and initial conditions
        model = test.LinearTestModel(order=2)
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(test.InitialCondition(0.0, geometries=[1]))
        model.initialConditions.append(test.InitialCondition(2.0, geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver(steps=[SolverStep(solver="cg", prec="gamg", condensation=True, symmetric=True)])
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("X", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

    def test_nonlinear(self):
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))
        p.mesher.setRefinement(5)

        # model: boundary and initial conditions
        model = test.NonlinearTestModel()
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(test.InitialCondition("x", geometries=[1]))
        model.initialConditions.append(test.InitialCondition("x", geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("X", data_number=1)
        for w in res:
            assert_array_almost_equal(w.data, np.sqrt(2 * w.x[:, 0]), decimal=2)
        c = np.array([0.5,0.6,0.7])
        assert_array_almost_equal(sol.eval("X", data_number=1, coords=c), np.sqrt(2*c))

    def test_smallGeom(self):
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1e-9, 0, 0))
        p.geometries.add(geometry.Line(1e-9, 0, 0, 2e-9, 0, 0))

        # model: boundary and initial conditions
        model = test.LinearTestModel()
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(test.InitialCondition(0.0, geometries=[1]))
        model.initialConditions.append(test.InitialCondition(2.0, geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("X", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data*1e-9, w.x[:, 0])

    def test_twoVars1(self):
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))

        # model: boundary and initial conditions
        model = test.TwoVariableTestModel()
        model.initialConditions.append(test.InitialCondition([1,0], geometries="all"))
        p.models.append(model)

        # solver
        stationary = TimeDependentSolver(0.001, 0.1)
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        t = np.linspace(0,0.1,101)
        x = [sol.eval("X", coords=0, data_number=i) for i in range(101)]
        y = [sol.eval("Y", coords=0, data_number=i) for i in range(101)]
        self.assert_array_almost_equal(x, (np.exp(-2*t)+1)/2, decimal=4)
        self.assert_array_almost_equal(y, (1-np.exp(-2*t))/2, decimal=4)

    def test_twoVars_step(self):
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))

        # model: boundary and initial conditions
        model = test.TwoVariableTestModel()
        model.initialConditions.append(test.InitialCondition([1,0], geometries="all"))
        p.models.append(model)

        # solver
        steps = [SolverStep(["X"]), SolverStep(["Y"], solver="cg", prec="gamg")]
        stationary = TimeDependentSolver(0.001, 0.1, steps=steps)
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        t = np.linspace(0,0.1,101)
        x = [sol.eval("X", coords=0, data_number=i) for i in range(101)]
        y = [sol.eval("Y", coords=0, data_number=i) for i in range(101)]
        self.assert_array_almost_equal(x, (np.exp(-2*t)+1)/2, decimal=4)
        self.assert_array_almost_equal(y, (1-np.exp(-2*t))/2, decimal=4)

    def test_twoVars_fix(self):
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))

        # model: boundary and initial conditions
        model1 = test.ExpTestModel()
        model1.initialConditions.append(test.InitialCondition("x", geometries="all"))
        p.models.append(model1)

        model2 = test.TdepFieldTestModel()
        model2.initialConditions.append(test.InitialCondition(0, geometries="all"))
        p.models.append(model2)

        # solver
        stationary = TimeDependentSolver(0.001, 0.1, steps=[SolverStep(["Y"])])
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        t = np.linspace(0,0.1,101)
        y = [sol.eval("Y", coords=0.5, data_number=i) for i in range(101)]
        self.assert_array_almost_equal(y, -0.5*t, decimal=4)

    def test_consts(self):
        p = FEMProject()
        p.parameters["a"] = 1
        p.parameters["b"] = "a+1"

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # model: boundary and initial conditions
        model = test.LinearTestModel()
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(test.InitialCondition(0.0, geometries=[1]))
        model.initialConditions.append(test.InitialCondition("b", geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("X", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

    def test_fields(self):
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # material
        param1 = test.UserDefinedParameters(x0 = 0)
        param2 = test.UserDefinedParameters(x0 = 1)
        p.materials.append(Material([param1], geometries=[1]))
        p.materials.append(Material([param2], geometries=[2]))

        # model: boundary and initial conditions
        model = test.LinearTestModel()
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(test.InitialCondition("x0", geometries="all"))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("X", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0]/2)

    def test_loadInitial_1d(self):
        if mpi.isRoot:
            os.makedirs("run1", exist_ok=True)
            os.makedirs("run2", exist_ok=True)
        mpi.wait()
        os.chdir("run1")

        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))

        # model: boundary and initial conditions
        model = test.LinearTestModel()
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(test.InitialCondition(0, geometries=[1]))
        model.initialConditions.append(test.InitialCondition(2, geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("X", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

        # second calculation
        os.chdir("../run2")
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 2, 0, 0))
        p.mesher.setRefinement(3)

        # solution fields
        p.solutionFields.add("x0", "../run1", "X")

        # model: boundary and initial conditions
        model = test.LinearTestModel()
        model.boundaryConditions.append(test.DirichletBoundary(True, geometries=[1]))
        model.initialConditions.append(test.InitialCondition("x0", geometries="all"))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(0.01, 0.1)
        p.solvers.append(solver)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("X", data_number=0)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])

    def test_tdepField(self):
        if mpi.isRoot:
            os.makedirs("run1", exist_ok=True)
            os.makedirs("run2", exist_ok=True)
        mpi.wait()
        os.chdir("run1")

        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))

        # model: boundary and initial conditions
        model = test.ExpTestModel()
        model.initialConditions.append(test.InitialCondition(1, geometries="all"))
        p.models.append(model)

        # solver
        stationary = TimeDependentSolver(0.001, 0.1)
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        t = np.linspace(0,0.1,101)
        x = [sol.eval("X", coords=0, data_number=i) for i in range(101)]
        self.assert_array_almost_equal(x, np.exp(-t), decimal=4)

        # second calculation
        os.chdir("../run2")
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))

        # solution fields
        p.solutionFields.add("X", "../run1", "X", index=None)

        # model: boundary and initial conditions
        model = test.TdepFieldTestModel()
        model.initialConditions.append(test.InitialCondition(0, geometries="all"))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(0.001, 0.1)
        p.solvers.append(solver)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        y = [sol.eval("Y", coords=0, data_number=i) for i in range(101)]
        self.assert_array_almost_equal(y, np.exp(-t)-1, decimal=4)

    def test_twoVars_grad(self):
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.mesher.setRefinement(5)

        # model: boundary and initial conditions
        model = test.TwoVarGradTestModel()
        model.initialConditions.append(test.InitialCondition(["x", 0], geometries="all"))
        p.models.append(model)

        # solver
        stationary = TimeDependentSolver(0.001, 0.1, steps=[SolverStep(["Y"])])
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        t = np.linspace(0,0.1,101)
        y = [sol.eval("Y", coords=0.5, data_number=i) for i in range(101)]
        self.assert_array_almost_equal(y, -t, decimal=4)

    def test_direct_solvers(self):
        self.solver("pardiso", None)
        self.solver("pardiso", None, cond=True)
        self.solver("pardisospd", None)
        self.solver("sparsecholesky", None)
        self.solver("masterinverse", None)
        self.solver("umfpack", None)

    def test_iterative_solvers(self):
        self.solver("cg", "gamg")
        self.solver("cg", "gamg", cond=True)
        self.solver("minres", "gamg")
        self.solver("symmlq", "gamg")
        self.solver("gmres", "gamg")
        self.solver("bcgs", "gamg")

    def test_preconditioners(self):
        self.solver("cg", "jacobi")
        self.solver("cg", "bjacobi")
        self.solver("cg", "ilu")
        self.solver("cg", "icc")
        self.solver("cg", "gamg")
        self.solver("cg", "sor")

    def solver(self, solver, prec, cond=False):
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))
        p.mesher.setRefinement(2)

        # model: boundary and initial conditions
        model = test.LinearTestModel(order=2)
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries=[1, 3]))
        model.initialConditions.append(test.InitialCondition(0.0, geometries=[1]))
        model.initialConditions.append(test.InitialCondition(2.0, geometries=[2]))
        p.models.append(model)

        # solver
        stationary = StationarySolver(steps=[SolverStep(solver=solver, prec=prec, condensation=cond)])
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("X", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0], decimal=2)

    def test_error(self):
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.mesher.setRefinement(4)

        # model: boundary and initial conditions
        model = test.NonlinearTestModel(order=1)
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries="all"))
        model.initialConditions.append(test.InitialCondition("x", geometries="all"))
        p.models.append(model)

        # solver
        stationary = StationarySolver()
        stationary.setAdaptiveMeshRefinement("X", 1500)
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("X", data_number=-1)
        for w in res:
            assert_array_almost_equal(w.data, np.sqrt(w.x[:, 0]), decimal=2)
    
    def test_random(self):
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.mesher.setRefinement(7)
        p.randomFields.add("R", "H1", tdep = True)

        # model: boundary and initial conditions
        model = test.RandomWalkModel()
        model.initialConditions.append(test.InitialCondition(0.0, geometries="all"))
        model.domainConditions.append(test.RandomForce("R", geometries="all"))
        p.models.append(model)

        # solver
        solver = TimeDependentSolver(0.5, 100)
        p.solvers.append(solver)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        w = sol.eval("X", data_number=-1)[0]
        self.assertTrue(abs(np.mean(w.data)) < 1)
        self.assertTrue(80 < np.var(w.data) < 120)

    def test_group(self):
        p = FEMProject()

        # geometry
        p.geometries.add(geometry.Line(0, 0, 0, 1, 0, 0))
        p.geometries.add(geometry.Line(1, 0, 0, 2, 0, 0))
        p.geometries.addGroup("Boundary", "edge", [1,3])
        p.geometries.addGroup("Domain", "D1", [1])
        p.geometries.addGroup("Domain", "D2", [2])

        # model: boundary and initial conditions
        model = test.LinearTestModel()
        model.boundaryConditions.append(test.DirichletBoundary([True], geometries="edge"))
        model.initialConditions.append(test.InitialCondition(0.0, geometries="D1"))
        model.initialConditions.append(test.InitialCondition(2.0, geometries="D2"))
        p.models.append(model)

        # solver
        stationary = StationarySolver(steps=[SolverStep()])
        p.solvers.append(stationary)

        # solve
        p.run()

        # solution
        sol = FEMSolution()
        res = sol.eval("X", data_number=1)
        for w in res:
            self.assert_array_almost_equal(w.data, w.x[:, 0])
