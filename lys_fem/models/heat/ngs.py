from ngsolve import grad, dx, ds

from lys_fem.ngs import NGSModel, util
from . import NeumannBoundary


class NGSHeatConductionModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model
        self._mat = mat

    def weakform(self, tnt):
        Cv, k = self._mat["C_v"], self._mat["k"]
        M, K, F = 0, 0, 0
        for eq in self._model.equations:
            u,v = tnt[eq.variableName]
            gu, gv = grad(u), grad(v)

            M += Cv * u * v * dx
            K += k * (gu * gv) * dx

            c = self._model.boundaryConditions.coef(NeumannBoundary)
            if c is not None:
                f = util.generateCoefficient(c, self.mesh)
                F += f * v * ds

        return 0, M, K, F
