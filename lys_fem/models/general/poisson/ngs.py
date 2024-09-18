from lys_fem.ngs import NGSModel, dx, grad
from .. import Source


class NGSPoissonModel(NGSModel):
    def __init__(self, model, mesh, vars, coef=None):
        super().__init__(model, mesh, vars, addVariables=True, order=2)
        self._model = model
        self._coef = coef

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            u, v = vars[eq.variableName]

            gu = grad(u)
            if self._coef is not None:
                gu = mat[self._coef].dot(gu)
            if "J" in mat:
                J = mat["J"]
                wf += J.dot(gu).dot(J.dot(grad(v)))/J.det() * dx
            else:
                wf += gu.dot(grad(v)) * dx

            f = self.coef(Source, "f")
            wf += f*v*dx
        return wf
    