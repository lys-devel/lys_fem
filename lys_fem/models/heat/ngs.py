from lys_fem.ngs import NGSModel, util
from lys_fem.ngs.util import grad, dx, ds
from . import NeumannBoundary


class NGSHeatConductionModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model
        self._mat = mat

    def weakform(self, vars):
        Cv, k = self._mat["C_v"], self._mat["k"]
        wf = util.NGSFunction()
        for eq in self._model.equations:
            var = vars[eq.variableName]
            u, v = var.trial, var.test

            wf += Cv * u.t * v * dx 
            wf += grad(u) * (k * grad(v)) * dx

            c = self._model.boundaryConditions.coef(NeumannBoundary)
            if c is not None:
                f = util.coef(c, self.mesh, name="f")
                wf -= f * v * ds
        return wf


    def weakform2(self, vars):
        Cv, k = self._mat["C_v"], self._mat["k"]
        M, K, F = 0, 0, 0
        for eq in self._model.equations:
            _, v = tnt[eq.variableName]
            u, ut, _ = vars[eq.variableName]
            gu, gv = grad(u), grad(v)

            M += Cv * ut * v * dx
            K += gu * (k * gv) * dx

            c = self._model.boundaryConditions.coef(NeumannBoundary)
            if c is not None:
                f = util.generateCoefficient(c, self.mesh)
                F += f * v * ds

        return 0, M, K, F
