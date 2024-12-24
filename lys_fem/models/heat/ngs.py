from lys_fem.ngs import NGSModel, grad, dx, ds
from . import NeumannBoundary


class NGSHeatConductionModel(NGSModel):
    def __init__(self, model, vars):
        super().__init__(model, vars, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        Cv, k = mat["C_v"], mat["k"]
        wf = 0
        for eq in self._model.equations:
            u, v = vars[eq.variableName]

            wf += Cv * u.t * v * dx + grad(u).dot(k.dot(grad(v))) * dx

            for n in self._model.boundaryConditions.get(NeumannBoundary):
                f = mat[n.values]
                wf -= f * v * ds(n.geometries)

        return wf

