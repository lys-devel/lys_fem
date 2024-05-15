from lys_fem.ngs import NGSModel, grad, dx


class NGSLinearTestModel(NGSModel):
    def __init__(self, model, mesh):
        super().__init__(model, mesh, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            u,v = vars[eq.variableName]
            wf += grad(u).dot(grad(v)) * dx
        return wf


class NGSNonlinearTestModel(NGSModel):
    def __init__(self, model, mesh):
        super().__init__(model, mesh, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            u,v = vars[eq.variableName]
            wf += u.value * grad(u).dot(grad(v)) * dx
        return wf

    @property
    def isNonlinear(self):
        return True