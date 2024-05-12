import itertools
import numpy as np

from ngsolve import dx, ds, grad
from ngsolve.fem import Einsum

from lys_fem.ngs import NGSModel, util
from . import ThermoelasticStress


class NGSElasticModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True, order=2)
        self._model = model
        self._mat = mat
        self._vdim = self._model.variableDimension()
        self._C = self.__getC(mesh.dim, self._mat["C"])

    def __getC(self, dim, C):
        res = np.zeros((dim,dim,dim,dim)).tolist()
        for i,j,k,l in itertools.product(range(dim),range(dim),range(dim),range(dim)):
            res[i][j][k][l] = C[self.__map(i,j), self.__map(k,l)]
        return util.generateCoefficient(res)
    
    def __map(self, i, j):
        if i == j:
            return i
        if (i == 0 and j == 1) or (i == 1 and j == 0):
            return 3
        if (i == 1 and j == 2) or (i == 2 and j == 1):
            return 4
        if (i == 0 and j == 2) or (i == 2 and j == 0):
            return 5

    def weakform(self, tnt, vars):
        C, rho = self._C, self._mat["rho"]
        M, K, F = 0, 0, 0
        for eq in self._model.equations:
            _,v = tnt[eq.variableName]
            u,ut,utt = vars[eq.variableName]
            gu, gv = grad(u), grad(v)
            
            M += rho * utt * v * dx
            K += gu * C * gv * dx if self._vdim==1 else Einsum("ij,ijkl,kl", gu, C, gv) * dx

            t0 = self._model.domainConditions.coef(ThermoelasticStress)
            if t0 is not None:
                alpha = self._mat["alpha"]
                T0 = util.generateCoefficient(t0, self._mesh)
                for te in self._model.domainConditions.get(ThermoelasticStress):
                    T, test_T = tnt[te.varName]
                    if self._vdim == 1:
                        K += T*C*alpha*gv*dx
                        F += T0*C*alpha*gv*dx
                    else:
                        beta = Einsum("ijkl,kl->ij", C, alpha)
                        K += T*Einsum("ij,ij", beta, gv)*dx
                        F += T0*Einsum("ij,ij", beta, gv)*dx
        return M, 0, K, F