from ngsolve import grad, dx, ds

from lys_fem.ngs import NGSModel, util
from . import NeumannBoundary


class NGSHeatConductionModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model
        self._mat = mat

    def weakform(self, tnt, vars):
        Cv, k = self._mat["C_v"], self._mat["k"]
        M, K, F = 0, 0, 0
        for eq in self._model.equations:
            _, v = tnt[eq.variableName]
            u, ut, _ = vars[eq.variableName]
            gu, gv = grad(u), grad(v)

            M += Cv * ut * v * dx
            K += k * (gu * gv) * dx

            c = self._model.boundaryConditions.coef(NeumannBoundary)
            if c is not None:
                f = util.generateCoefficient(c, self.mesh)
                F += f * v * ds

        return 0, M, K, F
