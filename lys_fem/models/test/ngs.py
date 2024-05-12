from lys_fem.ngs import NGSModel
from lys_fem.ngs.util import grad, dx, NGSFunction


class NGSLinearTestModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model

    def weakform(self, vars):
        wf = NGSFunction()
        for eq in self._model.equations:
            var = vars[eq.variableName]
            u,v = var.trial, var.test
            wf += grad(u).dot(grad(v)) * dx
        return wf


class NGSNonlinearTestModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model

    def weakform(self, vars):
        wf = NGSFunction()
        for eq in self._model.equations:
            var = vars[eq.variableName]
            u,v = var.trial, var.test
            wf += u0 * grad(u).dot(grad(v)) * dx
        return wf

    @property
    def isNonlinear(self):
        return True