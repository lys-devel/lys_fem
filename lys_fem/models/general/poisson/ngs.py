from lys_fem.ngs import NGSModel
from lys_fem.ngs.util import NGSFunction, dx, grad
from .. import Source


class NGSPoissonModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True, order=2)
        self._model = model
        self._mesh = mesh
        self._mat = mat

    def weakform(self, vars):
        wf = NGSFunction()
        for eq in self._model.equations:
            var = vars[eq.variableName]
            u, v = var.trial, var.test

            if "J" in self._mat:
                J, detJ = self._mat["J"], self._mat["detJ"]
                wf += J.dot(grad(u)).dot(J.dot(grad(v)))/detJ * dx
            else:
                wf += grad(u).dot(grad(v)) * dx

            f = self.coef(Source, "f")
            wf += f*v*dx
        return wf
    