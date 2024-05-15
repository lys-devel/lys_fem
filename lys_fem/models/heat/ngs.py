from lys_fem.ngs import NGSModel, grad, dx, ds
from . import NeumannBoundary


class NGSHeatConductionModel(NGSModel):
    def __init__(self, model, mesh):
        super().__init__(model, mesh, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        Cv, k = mat["C_v"], mat["k"]
        wf = 0
        for eq in self._model.equations:
            u, v = vars[eq.variableName]

            wf += Cv * u.t * v * dx + grad(u).dot(k.dot(grad(v))) * dx

            f = self.coef(NeumannBoundary, name="f")
            wf -= f * v * ds

        return wf

