import itertools
from .. import mfem


class ElasticModel(mfem.SecondOrderTimeDependentOperator):
    def __init__(self, model, mesh, mat, order=1):
        # Define a parallel finite element space on the parallel mesh.
        self._mesh = mesh
        self._dim = mesh.Dimension()
        self._fec = mfem.H1_FECollection(order, self._dim)
        self._fespace = mfem.FiniteElementSpace(mesh, self._fec, model.variableDimension(), mfem.Ordering.byVDIM)
        self._mat = mat
        self._dirichlet = []
        super().__init__(self._fespace.GetTrueVSize(), 0)

    @classmethod
    @property
    def name(self):
        return "Elasticity"

    def setInitialValue(self, x=None, xt=None):
        self._x_gf = mfem.GridFunction(self._fespace)
        self._xt_gf = mfem.GridFunction(self._fespace)
        if x is None:
            self._x_gf.Assign(0.0)
        else:
            self._x_gf.ProjectCoefficient(x)
        if xt is None:
            self._xt_gf.Assign(0.0)
        else:
            self._xt_gf.ProjectCoefficient(xt)

    def getInitialValue(self):
        return self._x_gf, self._xt_gf

    def setBoundaryStress(self, n_sigma):
        self._boundary_stress = n_sigma

    def setDirichletBoundary(self, boundaries):
        self._dirichlet = boundaries

    def assemble_m(self):
        m = mfem.ParBilinearForm(self._fespace)
        m.AddDomainIntegrator(mfem.VectorMassIntegrator())
        m.Assemble()
        return m

    def assemble_a(self):
        # 11. Set up the parallel bilinear form a(.,.) on the finite element space
        #     corresponding to the linear elasticity integrator with piece-wise
        #     constants coefficient lambda and mu.
        a = mfem.BilinearForm(self._fespace)
        a.AddDomainIntegrator(_ElasticityIntegrator(self._mat["Elasticity"]["C"]))
        a.Assemble()
        return a

    def assemble_b(self):
        f = mfem.VectorArrayCoefficient(self._mesh.Dimension())
        # for d in range(self._dim):
        #    pull_force = mfem.Vector(self._boundary_stress.T[d])
        #    f.Set(d, mfem.PWConstCoefficient(pull_force))

        b = mfem.LinearForm(self._fespace)
        # b.AddBoundaryIntegrator(mfem.VectorBoundaryLFIntegrator(f))
        b.Assemble()
        return b

    def essential_tdof_list(self):
        if len(self._dirichlet) == 0:
            return mfem.intArray()
        ess_bdr = mfem.intArray(self._mesh.bdr_attributes.Max())
        ess_bdr.Assign(0)  # ess_bdr = [0,0,0]
        for i in self._dirichlet:
            ess_bdr[i] = 1  # ess_bdr = [1,0,0]
        ess_tdof_list = mfem.intArray()
        self._fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)  # ess_tdof_list = list of dofs
        return ess_tdof_list


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
