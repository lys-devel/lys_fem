
import itertools

from mpi4py import MPI
import mfem.par as mfem


def generateMesh(file="beam-tri.mesh"):
    mesh = mfem.Mesh(file, 1, 1)
    for x in range(2):
        mesh.UniformRefinement()
    pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
    pmesh.UniformRefinement()
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


class ElasticModel:
    def __init__(self, mesh, dim, mat, order=1):
        # Define a parallel finite element space on the parallel mesh.
        self._mesh = mesh
        self._dim = mesh.Dimension()
        self._fec = mfem.H1_FECollection(order, dim)
        self._fespace = mfem.ParFiniteElementSpace(mesh, self._fec, dim, mfem.Ordering.byVDIM)
        self._mat = mat

    def solve(self, solver=None):
        if solver is None:
            solver = LinearSolver(2)
        x = mfem.GridFunction(self._fespace)
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
        ess_bdr = mfem.intArray(self._mesh.bdr_attributes.Max())
        ess_bdr.Assign(0)  # ess_bdr = [0,0,0]
        for i in self._dirichlet:
            ess_bdr[i] = 1  # ess_bdr = [1,0,0]
        ess_tdof_list = mfem.intArray()
        self._fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)  # ess_tdof_list = list of dofs
        return ess_tdof_list


class Material:
    def __init__(self, mesh):
        self._mesh = mesh

    def getLambda(self):
        lamb = mfem.Vector(self._mesh.attributes.Max())
        lamb[0] = 50
        lamb[1] = 1.0
        return mfem.PWConstCoefficient(lamb)

    def getMu(self):
        lamb = mfem.Vector(self._mesh.attributes.Max())
        lamb[0] = 50
        lamb[1] = 1.0
        return mfem.PWConstCoefficient(lamb)

    def getC(self, i, j, k, l):
        c = mfem.Vector(self._mesh.attributes.Max())
        res = 0
        if i == j and k == l:
            res += 1
        if i == k and j == l:
            res += 1
        if i == l and j == k:
            res += 1
        c[0] = res * 50
        c[1] = res
        return mfem.PWConstCoefficient(c)

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
