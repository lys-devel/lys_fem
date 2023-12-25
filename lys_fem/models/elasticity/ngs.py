import itertools
import numpy as np

from ngsolve import grad, dx, ds
from ngsolve.fem import Einsum

from lys_fem.ngs import NGSModel, dti, util


class NGSElasticModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh)
        self._model = model
        self._mat = mat
        self._vdim = self._model.variableDimension()
        self._C = self.__getC(mesh.dim, self._mat["C"])
        self.__generateVariables(mesh, model)

    def __generateVariables(self, mesh, model):
        dirichlet = util.generateDirichletCondition(model)
        init = util.generateDomainCoefficient(mesh, model.initialConditions)
        for eq in model.equations:
            self.addVariable(eq.variableName, eq.variableDimension, dirichlet, init, eq.geometries)

    @property
    def bilinearform(self):
        C, rho = self._C, self._mat["rho"]

        wf = 0
        for sp, eq in zip(self.spaces, self._model.equations):
            u,v =sp.TnT()
            gu, gv = grad(u), grad(v)
            if self._vdim == 1:
                wf += gu * C * gv * dx
            else:
                wf +=  Einsum("ij,ijkl,kl", gu, C, gv) * dx
        return wf
    
    @property
    def linearform(self):
        wf = 0
        for sp, u0, eq in zip(self.spaces, self.sol, self._model.equations):
            u,v =sp.TnT()
            wf += util.generateCoefficient([0]*self._vdim) * v * dx
        return wf

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