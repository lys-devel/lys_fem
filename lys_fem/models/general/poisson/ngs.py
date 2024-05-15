from lys_fem.ngs import NGSModel, dx, grad
from .. import Source


class NGSPoissonModel(NGSModel):
    def __init__(self, model, mesh):
        super().__init__(model, mesh, addVariables=True, order=2)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            u, v = vars[eq.variableName]

            if "J" in mat:
                J, detJ = mat["J"], mat["detJ"]
                wf += J.dot(grad(u)).dot(J.dot(grad(v)))/detJ * dx
            else:
                wf += grad(u).dot(grad(v)) * dx

            f = self.coef(Source, "f")
            wf += f*v*dx
        return wf
    