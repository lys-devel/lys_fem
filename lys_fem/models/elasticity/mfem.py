import itertools
import numpy as np
import sympy as sp

from lys_fem.mf import mfem, MFEMLinearModel, util, weakform, coef
from lys_fem.mf.weakform import grad, TrialFunction, TestFunction

from lys_fem.fem import NeumannBoundary

class MFEMElasticModel(MFEMLinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        self._mesh = mesh
        self._mat = mat
        self._model = model
        self._nvar = model.variableDimension()
        init = util.generateDomainCoefficient(mesh, model.initialConditions)
        self._u = weakform.TrialFunction("u", mesh, self.dirichletCondition, init, nvar=self._nvar)

    @property
    def trialFunctions(self):
        return [ui for ui in self._u]

    @property
    def weakform(self):
        t, dV, dS = sp.symbols("t,dV,dS")
        rho= sp.symbols("rho")
        C = self.__getC(self._mesh.SpaceDimension())

        u = self._u
        v = weakform.TestFunction(u)
        gu, gv = grad(u), grad(v)

        Cgv = sp.tensorcontraction(sp.tensorcontraction(sp.tensorproduct(C, gv), (2,4)), (2,3))
        D = sp.tensorcontraction(sp.tensorcontraction(sp.tensorproduct(gu, Cgv), (0,2)), (0,1))
        return (rho * u.diff(t).diff(t).dot(v) + D) * dV

    def __getC(self, dim):
        res = np.zeros((dim,dim,dim,dim)).tolist()
        for i,j,k,l in itertools.product(range(dim),range(dim),range(dim),range(dim)):
            res[i][j][k][l] = sp.Symbol("C"+str(self.__map(i,j))+str(self.__map(k,l)))
        return sp.Array(res)

    @property
    def coefficient(self):
        self._coefs = {"rho": self._mat["Elasticity"]["rho"]}
        C = self._mat["Elasticity"]["C"]
        for i in range(6):
            for j in range(6):
                self._coefs["C"+str(i+1)+str(j+1)] = C[i,j] 
        return self._coefs

    def __map(self, i, j):
        if i == j:
            return i + 1
        if (i == 0 and j == 1) or (i == 1 and j == 0):
            return 4
        if (i == 1 and j == 2) or (i == 2 and j == 1):
            return 5
        if (i == 0 and j == 2) or (i == 2 and j == 0):
            return 6  

class MFEMElasticModel_(MFEMLinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        self._mat = mat
        self._fec=mfem.H1_FECollection(1, mesh.Dimension())
        self._space =  mfem.FiniteElementSpace(mesh, self._fec, model.variableDimension(), mfem.Ordering.byVDIM)
        self._initialize(model)

    def _initialize(self, model):
        ess_tdof = self.essential_tdof_list(self._space)
        c = util.generateDomainCoefficient(self._space, model.initialConditions)
        self.x0 = util.initialValue(self._space, c)
        self.M = util.bilinearForm(self._space, ess_tdof, domainInteg=mfem.VectorMassIntegrator(self._mat["Elasticity"]["rho"]))
        self.K = util.bilinearForm(self._space, ess_tdof, domainInteg=_ElasticityIntegrator(self._mat["Elasticity"]["C"]))
        self.b = util.linearForm(self._space, ess_tdof, self.K, self.x0)

    @property
    def solution(self):
        gf = mfem.GridFunction(self._space)
        gf.SetFromTrueDofs(self.x0)
        return gf


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


