
import itertools

import mfem.par as mfem

from mpi4py import MPI
import gmsh
import numpy as np


def generateMesh(file):
    mesh = mfem.Mesh(file, 1, 1)
    if len([i for i in mesh.bdr_attributes]) == 0:  # For 1D mesh, we have to set boundary manually.
        # Load file by gmsh
        model = gmsh.model()
        model.add("Default")
        model.setCurrent("Default")
        gmsh.merge(file)

        # Get all boundary nodes
        s = set(np.array([model.mesh.getNodes(*obj, includeBoundary=True)[0][:2] for obj in model.getEntities(1)]).flatten())

        # Set the boundary nodes to mesh object.
        for v in s:
            mesh.AddBdrPoint(v, v)
        mesh.SetAttributes()
    pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
    return pmesh


class _ElasticityIntegrator(mfem.BilinearFormIntegrator):
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.dshape = mfem.DenseMatrix()
        self.gshape = mfem.DenseMatrix()

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
            for i, j, k, l in itertools.product(range(dim), range(dim), range(dim), range(dim)):
                C = self.C[i][j][k][l].Eval(trans, ip)
                for n, m in itertools.product(range(dof), range(dof)):
                    elmat[dof * i + m, dof * k + n] += C * w * self.gshape[n, l] * self.gshape[m, j]


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


class ElasticModel(mfem.SecondOrderTimeDependentOperator):
    def __init__(self, mesh, dim, mat, order=1):
        # Define a parallel finite element space on the parallel mesh.
        self._mesh = mesh
        self._dim = mesh.Dimension()
        self._fec = mfem.H1_FECollection(order, dim)
        self._fespace = mfem.ParFiniteElementSpace(mesh, self._fec, dim, mfem.Ordering.byVDIM)
        self._mat = mat
        self._dirichlet = []
        super().__init__(self._fespace.GetTrueVSize(), 0)

    def solve(self, solver=None):
        if solver is None:
            solver = LinearSolver(2)
        x = mfem.ParGridFunction(self._fespace)
        x.Assign(0.0)
        a = self._assemble_a()
        b = self._assemble_b()
        ess_tdof_list = self._essential_tdof_list()

        A = mfem.HypreParMatrix()
        B = mfem.Vector()
        X = mfem.Vector()
        a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)

        solver.apply(A, B, X)  # Solve AX=B
        a.RecoverFEMSolution(X, b, x)
        return x

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
        a.AddDomainIntegrator(_ElasticityIntegrator(self._mat.calcMatrix()))
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

    def setInitialValue(self, x=None, xt=None):
        self._x_gf = mfem.ParGridFunction(self._fespace)
        self._xt_gf = mfem.ParGridFunction(self._fespace)
        self._x_gf.Assign(0.0)
        self._xt_gf.Assign(0.0)

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


class Material:
    def __init__(self, mesh):
        self._mesh = mesh

    def getC(self, i, j, k, l):
        res = 0
        if i == j and k == l:
            res += 1
        if i == k and j == l:
            res += 1
        if i == l and j == k:
            res += 1
        return mfem.ConstantCoefficient(res)

    def calcMatrix(self):
        return [[[[self.getC(i, j, k, l) for l in range(3)] for k in range(3)] for j in range(3)] for i in range(3)]


class LinearSolver:
    def __init__(self, dim):
        self._dim = dim

    def apply(self, A, B, X):
        self._amg = mfem.HypreBoomerAMG(A)
        self._amg.SetSystemsOptions(self._dim)
        self._pcg = mfem.HyprePCG(A)
        self._pcg.SetTol(1e-8)
        self._pcg.SetMaxIter(500)
        self._pcg.SetPrintLevel(2)
        self._pcg.SetPreconditioner(self._amg)
        self._pcg.Mult(B, X)
