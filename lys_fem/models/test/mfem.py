from lys_fem.mf import mfem, util, MFEMLinearModel, MFEMNonlinearModel
from lys_fem.mf import MFEMLinearModel, util, weakform
from lys_fem.mf.weakform import grad, dV


class MFEMLinearTestModel(MFEMLinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        self._mesh = mesh
        self._mat = mat
        self._model = model
        self._u = weakform.TrialFunction("x", mesh, self.dirichletCondition[0], util.generateDomainCoefficient(mesh, model.initialConditions, 0))

    @property
    def trialFunctions(self):
        return [self._u]

    @property
    def weakform(self):
        u = self._u
        v = weakform.TestFunction(u)
        return grad(u).dot(grad(v)) * dV

    @property
    def coefficient(self):
        return {}

class MFEMNonlinearTestModel(MFEMNonlinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        self._mesh = mesh
        self._mat = mat
        self._model = model
        self._u = weakform.TrialFunction("x", mesh, self.dirichletCondition[0], util.generateDomainCoefficient(mesh, model.initialConditions, 0))

    @property
    def trialFunctions(self):
        return [self._u]

    @property
    def weakform(self):
        u = self._u
        v = weakform.TestFunction(u)
        return u*grad(u).dot(grad(v)) * dV

    @property
    def coefficient(self):
        return {}

