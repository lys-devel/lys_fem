import itertools
import numpy as np
from .. import mfem


class ElasticModel(mfem.SecondOrderTimeDependentOperator):
    def __init__(self, fec, model, mesh, mat):
        # Define a parallel finite element space on the parallel mesh.
        self._mesh = mesh
        self._dim = mesh.Dimension()
        self._fec = fec
        self._fespace = mfem.FiniteElementSpace(mesh, self._fec, model.variableDimension(), mfem.Ordering.byVDIM)
        self._mat = mat
        self._bdr_stress = None
        self._dirichlet = None
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

    def setBoundaryStress(self, stress):
        self._bdr_stress = stress

    def setDirichletBoundary(self, dirichlet_dict):
        self._dirichlet = dirichlet_dict

    def assemble_m(self):
        m = mfem.BilinearForm(self._fespace)
        m.AddDomainIntegrator(mfem.VectorMassIntegrator())
        m.Assemble()
        return m

    def assemble_a(self):
        a = mfem.BilinearForm(self._fespace)
        i = mfem.ElasticityIntegrator(mfem.ConstantCoefficient(1), mfem.ConstantCoefficient(1))
        # a.AddDomainIntegrator()
        a.AddDomainIntegrator(_ElasticityIntegrator(self._mat["Elasticity"]["C"], i))
        a.Assemble()
        return a

    def assemble_b(self):
        b = mfem.LinearForm(self._fespace)
        if self._bdr_stress is not None:
            b.AddBoundaryIntegrator(mfem.VectorBoundaryLFIntegrator(self._bdr_stress))
        b.Assemble()
        return b

    def essential_tdof_list(self):
        if self._dirichlet is None:
            return mfem.intArray()
        res = []
        for axis, b in self._dirichlet.items():
            ess_bdr = mfem.intArray(self._mesh.bdr_attributes.Max())
            ess_bdr.Assign(0)
            for i in b:
                ess_bdr[i - 1] = 1
            ess_tdof_list = mfem.intArray()
            self._fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list, axis)
            res.extend([i for i in ess_tdof_list])
        return mfem.intArray(res)


class _ElasticityIntegrator(mfem.BilinearFormIntegrator):
    def __init__(self, C, i):
        super().__init__()
        self.C = C
        self.dshape = mfem.DenseMatrix()
        self.gshape = mfem.DenseMatrix()
        self._Ce = mfem.DenseMatrix()
        self._i = i

    def AssembleElementMatrix(self, el, trans, elmat):
        # Initialize variables
        dof, dim = el.GetDof(), el.GetDim()
        self.dshape.SetSize(dof, dim)
        self.gshape.SetSize(dof, dim)
        elmat.SetSize(dof * dim)
        elmat.Assign(0.0)

        ir = mfem.IntRules.Get(el.GetGeomType(), 2 * el.GetOrder())
        res = np.zeros((dim, dof, dim, dof), dtype=float)
        for p in range(ir.GetNPoints()):
            ip = ir.IntPoint(p)
            trans.SetIntPoint(ip)
            w = ip.weight * trans.Weight()

            # calculate gshape (grad of shape function)
            el.CalcDShape(ip, self.dshape)
            mfem.Mult(self.dshape, trans.InverseJacobian(), self.gshape)

            C = self.__getC(trans, ip, dim)
            g = w * self.__getGShape(self.gshape, dof, dim)
            res += np.tensordot(np.tensordot(C, g, (3, 1)), g, (1, 1)).transpose(0, 3, 1, 2)  # equivalent to eingsum ijkl,nl,mj->imkn
        elmat.Add(1, mfem.DenseMatrix(res.reshape((dim * dof, dim * dof))))

    def __getC(self, trans, ip, dim):
        self.C.Eval(self._Ce, trans, ip)
        res = np.empty((dim, dim, dim, dim))
        for i, j, k, l in itertools.product(range(dim), range(dim), range(dim), range(dim)):
            p, q = self.__map(i, j), self.__map(k, l)
            res[i, j, k, l] = self._Ce[p, q]
        return res

    def __getGShape(self, gshape, dof, dim):
        res = np.empty((dof, dim))
        for i, j in itertools.product(range(dof), range(dim)):
            res[i, j] = gshape[i, j]
        return res

    def __map(self, i, j):
        if i == j:
            return i
        if (i == 0 and j == 1) or (i == 1 and j == 0):
            return 3
        if (i == 1 and j == 2) or (i == 2 and j == 1):
            return 4
        if (i == 0 and j == 2) or (i == 2 and j == 0):
            return 5


# class StressIntegrator(mfem.VectorDomainLFIntegrator):
class StressIntegrator(mfem.LinearFormIntegrator):
    def __init__(self):
        # super().__init__()
        # self.divshape = mfem.Vector()
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
