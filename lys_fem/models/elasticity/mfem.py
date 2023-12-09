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
        init = util.generateDomainCoefficient(mesh, model.initialConditions, sp.Matrix([0]*self._nvar))
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
        if (i == 0 and j =