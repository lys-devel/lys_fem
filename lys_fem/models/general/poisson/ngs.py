from ngsolve import grad, dx, ds, x, y, z

from .. import Source
from lys_fem.ngs import NGSModel, util


class NGSPoissonModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True, order=2)
        self._model = model
        self._mesh = mesh
        self._mat = mat

    def bilinearform(self, tnt, sols):
        wf = 0
        if "J" in self._mat:
            J = self._mat["J"]
            detJ = self._mat["detJ"]
        else:
            J = 1
            detJ = 1
        for eq in self._model.equations:
            u,v = tnt[eq.variableName]
            wf += ((J*grad(u))*(J*grad(v)))/detJ * dx
        return wf
    
    def linearform(self, tnt, sols):
        wf = util.generateCoefficient(0) * dx
        for eq in self._model.equations:
            u,v = tnt[eq.variableName]
            c = self._model.domainConditions.coef(Source)
            if c is not None:
                f = util.generateCoefficient(c, self._mesh)
                wf += -f*v*dx
        return wf
