import sympy as sp
from lys_fem.fem import NeumannBoundary
from lys_fem.mf import MFEMLinearModel, util, weakform, coef
from lys_fem.mf.weakform import grad


class MFEMHeatConductionModel(MFEMLinearModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        self._mesh = mesh
        self._mat = mat
        self._model = model
        self._u = weakform.TrialFunction("T", mesh, self.dirichletCondition[0], util.generateDomainCoefficient(mesh, model.initialConditions))

    @property
    def trialFunctions(self):
        return [self._u]

    @property
    def weakform(self):
        t, dV, dS = sp.symbols("t,dV,dS")
        Cv, k, f = sp.symbols("Cv,k,nDT")

        u = self._u
        v = weakform.TestFunction(u)
        gu, gv = grad(u), grad(v)

        return (Cv * u.diff(t) * v + k * gu.dot(gv)) * dV  + f * v * dS

    @property
    def coefficient(self):
        coefs = {"Cv": self._mat["Heat Conduction"]["C_v"], "k": self._mat["Heat Conduction"]["k"]}

        # neumann boundary condition
        neumann = [b for b in self._model.boundaryConditions if isinstance(b, NeumannBoundary)]
        if len(neumann) == 0:
            coefs["nDT"] = coef.ScalarCoef({}, self._mesh.SpaceDimension())
        else:
            coefs["nDT"]=util.generateSurfaceCoefficient(self._mesh, neumann)[0]
        return coefs
    