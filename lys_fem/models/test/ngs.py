from ngsolve import grad, dx, ds
from lys_fem.ngs import NGSModel, util


class NGSLinearTestModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model

    @property
    def bilinearform(self):
        wf = 0
        for sp, eq in zip(self.spaces, self._model.equations):
            u,v =sp.TnT()
            gu, gv = grad(u), grad(v)
            wf += (gu*gv) * dx
        return wf
    
    @property
    def linearform(self):
        wf = 0
        for sp, eq in zip(self.spaces, self._model.equations):
            u,v =sp.TnT()
            wf += util.generateCoefficient(0) * dx
        return wf

class NGSNonlinearTestModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model

    @property
    def bilinearform(self):
        wf = 0
        for sp, eq in zip(self.spaces, self._model.equations):
            u,v = sp.TnT()
            gu, gv = grad(u), grad(v)
            wf += u * (gu*gv) * dx
        return wf
    
    @property
    def linearform(self):
        wf = 0
        for sp, eq in zip(self.spaces, self._model.equations):
            u,v = sp.TnT()
            wf += util.generateCoefficient(0) * dx
        return wf

    @property
    def isNonlinear(self):
        return True