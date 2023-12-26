import sympy as sp
#from lys_fem.fem import NeumannBoundary
from lys_fem.mf import MFEMModel, util, weakform
from lys_fem.mf.weakform import grad, t, dV, dS


class MFEMHeatConductionModel(MFEMModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        self._mesh = mesh
        self._mat = mat
        self._model = model
        self._u = weakform.TrialFunction("T", mesh, self.dirichletCondition[0], util.generateDomainCoefficient(mesh, model.initialConditions, 0))

    @property
    def trialFunctions(self):
        return [self._u]

    @property
    def weakform(self):
        u = self._u
        v = weakform.TestFunction(u)
        gu, gv = grad(u), grad(v)
        Cv, k = sp.symbols("Cv,k")

        wf = (Cv * u.diff(t) * v + k * gu.dot(gv)) * dV

        if self._model.boundaryConditions.have(NeumannBoundary):
            f = sp.Symbol("nDT")
            wf += - f * v * dS
        return wf

    @property
    def coefficient(self):
        coefs = {"Cv": self._mat["Heat Conduction"]["C_v"], "k": self._mat["Heat Conduction"]["k"]}

        # neumann boundary condition
        if self._model.boundaryConditions.have(NeumannBoundary):
            neumann = self._model.boundaryConditions.get(NeumannBoundary)
            coefs["nDT"]=util.generateSurfaceCoefficient(self._mesh, neumann, default=0)
        return coefs
    