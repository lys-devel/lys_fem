from lys_fem.ngs import NGSModel, grad, dx, ds
from . import NeumannBoundary


class NGSHeatConductionModel(NGSModel):
    def __init__(self, model):
        super().__init__(model)
        self._model = model

    def weakform(self, vars, mat):
        Cv, k = mat["C_v"], mat["k"]
        wf = 0
        u, v = vars[self._model.variableName]

        wf += Cv * u.t * v * dx + grad(u).dot(k.dot(grad(v))) * dx

        for n in self._model.boundaryConditions.get(NeumannBoundary):
            f = mat[n.values]
            wf -= f * v * ds(n.geometries)

        return wf

