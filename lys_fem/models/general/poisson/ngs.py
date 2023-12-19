from ngsolve import grad, dx, ds

from lys_fem.fem import Source
from lys_fem.ngs import NGSModel, util


class NGSPoissonModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh)
        self._model = model
        self._mesh = mesh

    @property
    def bilinearform(self):
        u,v =self.TnT()
        wf = (grad(u)*grad(v)) * dx
        return wf
    
    @property
    def linearform(self):
        u,v =self.TnT()
        wf = util.generateCoefficient(0) * dx
        if self._model.domainConditions.have(Source):
            source = self._model.domainConditions.get(Source)
            f = util.generateDomainCoefficient(self._mesh, source)
            wf += -f*v*dx
        return wf
