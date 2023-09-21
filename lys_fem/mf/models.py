
import itertools
import numpy as np

import mfem.par as mfem
from mpi4py import MPI

from .coef import generateCoefficient


def generateModel(fem, geom, mesh, mat):
    nvar = fem.models[0].variableDimension()
    model = ElasticModel(mesh, nvar, mat)
    attrs = [tag for dim, tag in geom.getEntities(fem.dimension)]
    coefs = {}
    for init in fem.models[0].initialConditions:
        for d in attrs if init.domains == "all" else init.domains:
            coefs[d] = init.values
    c = generateCoefficient(coefs, geom)
    model.setInitialValue(c)
    return model


class ElasticModel(mfem.SecondOrderTimeDependentOperator):
    def __init__(self, mesh, nvar, mat, order=1):
        # Define a parallel finite element space on the parallel mesh.
        self._mesh = mesh
        self._dim = mesh.Dimension()
        self._fec = mfem.H1_FECollection(order, nvar)
        self._fespace = mfem.ParFiniteElementSpace(mesh, self._fec, nvar, mfem.Ordering.byVDIM)
        self._mat = mat
        self._dirichlet = []
        super().__init__(self._fespace.GetTrueVSize(), 0)

    def setInitialValue(self, x=None, xt=None):
        self._x_gf = mfem.ParGridFunction(self._fespace)
        self._xt_gf = mfem.ParGridFunction(self._fespace)
        if x is None:
            self._x_gf.Assign(0.0)
        else:
            self._x_gf.ProjectCoefficient(x)
        if xt is None:
            self._xt_gf.Assign(0.0)
        else:
            self._xt_gf.ProjectCoefficient(xt)

    def setBoundaryStress(self, n_sigma):
        self._boundary_stress = n_sigma

    def setDirichletBoundary(self, boundaries):
        self._dirichlet = boundaries

    def _assemble_m(self):
        m = mfem.ParBilinearForm(self._fespace)
        m.AddDomainIntegrator(mfem.VectorMassIntegrator())
        m.Assemble()
        return m

    def _assemble_a(self):
        # 11. Set up the parallel bilinear form a(.,.) on the finite element space
        #     corresponding to the linear elasticity integrator with piece-wise
        #     constants coefficient lambda and mu.
        a = mfem.ParBilinearForm(self._fespace)
        a.AddDomainIntegrator(_ElasticityIntegrator(self._mat["Elasticity"]["C"]))
        a.Assemble()
        return a

    def _assemble_b(self):
        f = mfem.VectorArrayCoefficient(self._mesh.Dimension())
        for d in range(self._dim):
            pull_force = mfem.Vector(self._boundary_stress.T[d])
            f.Set(d, mfem.PWConstCoefficient(pull_force))

        b = mfem.ParLinearForm(self._fespace)
        b.AddBoundaryIntegrator(mfem.VectorBoundaryLFIntegrator(f))
        b.Assemble()
        return b

    def _essential_tdof_list(self):
        if len(self._dirichlet) == 0:
            return mfem.intArray()
        ess_bdr = mfem.intArray(self._mesh.bdr_attributes.Max())
        ess_bdr.Assign(0)  # ess_bdr = [0,0,0]
        for i in self._dirichlet:
            ess_bdr[i] = 1  # ess_bdr = [1,0,0]
        ess_tdof_list = mfem.intArray()
        self._fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)  # ess_tdof_list = list of dofs
        return ess_tdof_list

    def solve(self):
        x = mfem.ParGridFunction(self._fespace)
        x.Assign(0.0)
        a = self._assemble_a()
        b = self._assemble_b()
        ess_tdof_list = self._essential_tdof_list()

        A = mfem.HypreParMatrix()
        B = mfem.Vector()
        X = mfem.Vector()
        a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)

        solver = self._getDefaultSolver(A)
        solver.Mult(B, X)  # Solve AX=B
        a.RecoverFEMSolution(X, b, x)
        return x

    def step(self, t, dt):
        if t == 0:
            self._ode_solver = mfem.GeneralizedAlpha2Solver(1)
            self._ode_solver.Init(self)

            self._x = mfem.Vector()
            self._x_gf.GetTrueDofs(self._x)

            self._xt = mfem.Vector()  # dx/dt
            self._xt_gf.GetTrueDofs(self._xt)

            self._K_op = self._assemble_a()
            self._M_op = self._assemble_m()
            ess_tdof_list = self._essential_tdof_list()

            self._K0 = mfem.HypreParMatrix()
            self._K = mfem.HypreParMatrix()
            dummy = mfem.intArray()
            self._K_op.FormSystemMatrix(dummy, self._K0)
            self._K_op.FormSystemMatrix(ess_tdof_list, self._K)

            self._M = mfem.HypreParMatrix()
            self._M_op.FormSystemMatrix(ess_tdof_list, self._M)

            self._M_solver, self._M_prec = self._getDefaultSolver(self._M)
            self._T = None

        return self._ode_solver.Step(self._x, self._xt, t, dt)

    def Mult(self, u, du_dt, d2udt2):
        # Compute d2udt2 = M^{-1}*-K(u)
        z = mfem.Vector(u.Size())
        self._K.Mult(u, z)
        z.Neg()  # z = -z
        self._M_solver.Mult(z, d2udt2)

    def ImplicitSolve(self, fac0, fac1, u, dudt, d2udt2):
        # Solve the equation: d2udt2 = M^{-1}*[-K(u + fac0*d2udt2)]
        if self._T is None:
            self._T = mfem.Add(1.0, self._M, fac0, self._K)
            self._T_solver, self._T_prec = self._getDefaultSolver(self._T)
        z = mfem.Vector(u.Size())
        self._K0.Mult(u, z)
        z.Neg()

        # iterate over Array<int> :D
        for j in self._essential_tdof_list():
            z[j] = 0.0

        self._T_solver.Mult(z, d2udt2)

    def _getDefaultSolver(self, A, rel_tol=1e-8):
        prec = mfem.HypreBoomerAMG(A)
        solver = mfem.CGSolver(MPI.COMM_WORLD)
        solver.iterative_mode = False
        solver.SetRelTol(rel_tol)
        solver.SetAbsTol(0.0)
        solver.SetMaxIter(100)
        solver.SetPrintLevel(0)
        solver.SetPreconditioner(prec)
        solver.SetOperator(A)
        return solver, prec


class _ElasticityIntegrator(mfem.BilinearFormIntegrator):
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.dshape = mfem.DenseMatrix()
        self.gshape = mfem.DenseMatrix()
        self._Ce = mfem.DenseMatrix()

    def AssembleElementMatrix(self, el, trans, elmat):
        # Initialize variables
        dof, dim = el.GetDof(), el.GetDim()
        self.dshape.SetSize(dof, dim)
        self.gshape.SetSize(dof, dim)
        elmat.SetSize(dof * dim)
        elmat.Assign(0.0)

        ir = mfem.IntRules.Get(el.GetGeomType(), 2 * el.GetOrder())
        for p in range(ir.GetNPoints()):
            ip = ir.IntPoint(p)
            trans.SetIntPoint(ip)
            w = ip.weight * trans.Weight()

            # calculate gshape (grad of shape function)
            el.CalcDShape(ip, self.dshape)
            mfem.Mult(self.dshape, trans.InverseJacobian(), self.gshape)

            # calculate element matrix
            self.C.Eval(self._Ce, trans, ip)
            for i, j, k, l in itertools.product(range(dim), range(dim), range(dim), range(dim)):
                p, q = self.__map(i, j), self.__map(k, l)
                for n, m in itertools.product(range(dof), range(dof)):
                    elmat[dof * i + m, dof * k + n] += self._Ce[p, q] * w * self.gshape[n, l] * self.gshape[m, j]

    def __map(self, i, j):
        if i == j:
            return i
        if (i == 0 and j == 1) or (i == 1 and j == 0):
            return 4
        if (i == 1 and j == 2) or (i == 2 and j == 1):
            return 5
        if (i == 0 and j == 2) or (i == 2 and j == 0):
            return 6


# class StressIntegrator(mfem.VectorDomainLFIntegrator):
class StressIntegrator(mfem.LinearFormIntegrator):
    def __init__(self):
        # super().__init__()
        #self.divshape = mfem.Vector()
        pass

    def AssembleRHSElementVect(self, el, Tr, elvect):
        return
        print("---------------------assemble---------------------------")
        dof = el.GetDof()
        print(dof)
        return
        self.divshape.SetSize(dof)
        elvect.SetSize(dof)
        elvect.Assign(0.0)

        ir = mfem.IntRules.Get(el.GetGeomType(), 2 * el.GetOrder())
        for p in range(ir.GetNPoints()):
            ip = ir.IntPoint(p)
            Tr.SetIntPoint(ip)

            val = Tr.Weight() * Q.Eval(Tr, ip)
            el.CalcPhysDivShape(Tr, self.divshape)
            mfem.add(elvect, ip.weight * val, self.divshape, elvect)
