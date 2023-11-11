import itertools
import numpy as np

from lys_fem.mf import mfem, MFEMLinearModel


class MFEMElasticModel(MFEMLinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(fec, model, mesh)
        self._mat = mat

    def assemble_m(self):
        m = mfem.BilinearForm(self.space)
        m.AddDomainIntegrator(mfem.VectorMassIntegrator(self._mat["Elasticity"]["rho"]))
        m.Assemble()
        return m

    def assemble_a(self):
        a = mfem.BilinearForm(self.space)
        #a.AddDomainIntegrator(mfem.ElasticityIntegrator(mfem.ConstantCoefficient(1), mfem.ConstantCoefficient(1)))
        a.AddDomainIntegrator(_ElasticityIntegrator(self._mat["Elasticity"]["C"]))
        a.Assemble()
        return a


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
