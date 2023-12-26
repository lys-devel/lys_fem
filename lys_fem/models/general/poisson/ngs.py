from ngsolve import grad, dx, ds

from .. import Source
from lys_fem.ngs import NGSModel, util


class NGSPoissonModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh)
        self._model = model
        self._mesh = mesh
        self.__generateVariables(mesh, model)

    def __generateVariables(self, mesh, model):
        dirichlet = util.generateDirichletCondition(model)
        init = util.generateDomainCoefficient(mesh, model.initialConditions)
        for eq in model.equations:
            self.addVariable(eq.variableName, eq.variableDimension, dirichlet, init, eq.geometries)

    @property
    def bilinearform(self):
        wf = 0
        for sp, eq in zip(self.spaces, self._model.equations):
            u,v = sp.TnT()
            wf += (grad(u)*grad(v)) * dx
        return wf
    
    @property
    def linearform(self):
        wf = util.generateCoefficient(0) * dx
        wf = 0
        for sp, eq in zip(self.spaces, self._model.equations):
            u,v =sp.TnT()
            if self._model.domainConditions.have(Source):
                source = self._model.domainConditions.get(Source)
                f = util.generateGeometryCoefficient(self._mesh, source)
                wf += -f*v*dx
        return wf
