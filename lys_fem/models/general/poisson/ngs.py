from ngsolve import grad, dx, ds

from .. import Source
from lys_fem.ngs import NGSModel, util


class NGSPoissonModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True, order=2)
        self._model = model
        self._mesh = mesh
        self._mat = mat

    def weakform(self, tnt, vars):
        J, detJ = self.__createJ()
        K, F = 0, 0
        for eq in self._model.equations:
            _,v = tnt[eq.variableName]
            u,ut,utt = vars[eq.variableName]
            K += (J*grad(u))*(J*grad(v))/detJ * dx

            c = self._model.domainConditions.coef(Source)
            if c is not None:
                f = util.generateCoefficient(c, self._mesh)
                F += -f*v*dx
        return 0, 0, K, F
    
    def __createJ(self):
        if "J" in self._mat:
            J = self._mat["J"]
            detJ = self._mat["detJ"]
        else:
            J = 1
            detJ = 1
        return J, detJ
