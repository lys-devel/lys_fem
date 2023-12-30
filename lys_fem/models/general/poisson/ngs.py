from ngsolve import grad, dx, ds, x, y, z

from .. import Source
from lys_fem.ngs import NGSModel, util


class NGSPoissonModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model
        self._mesh = mesh
        self._mat = mat

    @property
    def bilinearform(self):
        wf = 0
        if "J_T" in self._mat:
            J = self._mat["J_T"]
            detJ = J[0,0]*J[1,1]*J[2,2] + J[0,1]*J[1,2]*J[2,0] + J[0,2]*J[1,0]*J[2,1] - J[0,2]*J[1,1]*J[2,0] - J[0,1]*J[1,0]*J[2,2] - J[0,0]*J[1,2]*J[2,1]
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
        wf = 0
        for sp, eq in zip(self.spaces, self._model.equations):
            u,v =sp.TnT()
            if self._model.domainConditions.have(Source):
                source = self._model.domainConditions.get(Source)
                f = util.generateGeometryCoefficient(self._mesh, source)
                wf += -f*v*dx
        return wf
