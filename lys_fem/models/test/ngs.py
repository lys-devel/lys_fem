from ngsolve import grad, dx, ds
from lys_fem.ngs import NGSModel, util


class NGSLinearTestModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model

    def bilinearform(self, tnt, sols):
        wf = 0
        for eq in self._model.equations:
            u,v = tnt[eq.variableName]
            gu, gv = grad(u), grad(v)
            wf += (gu*gv) * dx
        return wf
    
    def linearform(self, tnt, sols):
        return util.generateCoefficient(0) * dx

class NGSNonlinearTestModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model

    def bilinearform(self, tnt, sols):
        wf = 0
        for eq in self._model.equations:
            u,v = tnt[eq.variableName]
            gu, gv = grad(u), grad(v)
            wf += u * (gu*gv) * dx
        return wf
    
    def linearform(self, tnt, sols):
        wf = util.generateCoefficient(0) * dx
        return wf

    @property
    def isNonlinear(self):
        return True