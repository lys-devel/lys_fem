from ngsolve import grad, dx, ds
from lys_fem.ngs import NGSModel, util


class NGSLinearTestModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model

    def weakform(self, tnt, vars):
        K, F = 0, 0
        for eq in self._model.equations:
            u,v = tnt[eq.variableName]
            gu, gv = grad(u), grad(v)
            K += (gu*gv) * dx
        return 0, 0, K, 0


class NGSNonlinearTestModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True)
        self._model = model

    def weakform(self, tnt, vars):
        K, F = 0, 0
        for eq in self._model.equations:
            u0,v = tnt[eq.variableName]
            u, ut, utt = vars[eq.variableName]
            gu, gv = grad(u0), grad(v)
            K += u0 * (gu*gv) * dx
        return 0,0,K,F

    @property
    def isNonlinear(self):
        return True