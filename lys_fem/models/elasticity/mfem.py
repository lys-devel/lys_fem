import itertools
import numpy as np

from lys_fem.mf import mfem, MFEMLinearModel


class MFEMElasticModel(MFEMLinearModel):
    def __init__(self, model, mesh, mat):
        self._mat = mat
        super().__init__(mfem.H1_FECollection(1, mesh.Dimension()), model, mesh)
        self._initialize(model)

    def _initialize(self, model):
        ess_tdof = self.essential_tdof_list()
        c = self.generateDomainCoefficient(self.space, model.initialConditions)
        self._X0 = self.initialValue(self.space, c)
        self._M = self.bilinearForm(self.space, ess_tdof, mfem.VectorMassIntegrator(self._mat["Elasticity"]["rho"]))
        self._K = self.bilinearForm(self.space, ess_tdof, _ElasticityIntegrator(self._mat["Elasticity"]["C"]))
        self._B = self.linearForm(self.space, ess_tdof, self._K, self.x0, [], [])

    def RecoverFEMSolution(self, X):
        self._X0 = X
        self.gf = mfem.GridFunction(self.space)
        self.gf.SetFromTrueDofs(X)
        return self.gf

    @property
    def M(self):
        return self._M
    
    @property
    def K(self):
        return self._K
    
    @property
    def b(self):
        return self._B

    @property
    def x0(self):
        return self._X0


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


