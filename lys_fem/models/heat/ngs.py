from lys_fem.ngs import NGSModel
from lys_fem.ngs.util import grad, dx, ds, NGSFunction
from . import NeumannBoundary


class NGSHeatConductionModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model
        self._mat = mat

    def weakform(self, vars):
        Cv, k = self._mat["C_v"], self._mat["k"]
        wf = NGSFunction()
        for eq in self._model.equations:
            var = vars[eq.variableName]
            u, v = var.trial, var.test

            wf += Cv * u.t * v * dx + grad(u).dot(k.dot(grad(v))) * dx

            f = self.coef(NeumannBoundary, name="f")
            wf -= f * v * ds

        return wf

