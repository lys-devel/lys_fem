from ngsolve import grad, dx, ds, x, y, z

from .. import Source
from lys_fem.ngs import NGSModel, util


class NGSPoissonModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True, order=2)
        self._model = model
        self._mesh = mesh
        self._mat = mat

    @property
    def bilinearform(self):
        wf = 0
        if "J" in self._mat:
            J = self._mat["J"]
            detJ = self._mat["detJ"]
        else:
            J = 1
            detJ = 1
        for sp, eq in zip(self.spaces, self._model.equations):
            u,v = sp.TnT()
            wf += ((J*grad(u))*(J*grad(v)))/detJ * dx
        return wf
    
    @property
    def linearform(self):
        wf = util.generateCoefficient(0) * dx
        for sp, eq in zip(self.spaces, self._model.equations):
            u,v =sp.TnT()
            if self._model.domainConditions.have(Source):
                source = self._model.domainConditions.get(Source)
                f = util.generateGeometryCoefficient(self._mesh, source)
                wf += -f*v*dx
        return wf
