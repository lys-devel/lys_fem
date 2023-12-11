import sympy as sp
from lys_fem.mf import MFEMModel, util, weakform, mfem
from lys_fem.mf.weakform import t,dV,dS


class MFEMLLGModel(MFEMModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model)
        self._mesh = mesh
        self._mat = mat
        self._model = model
        self._m = weakform.TrialFunction("m", mesh, self.dirichletCondition, util.generateDomainCoefficient(mesh, model.initialConditions, [1,0,0]), nvar=3)
        self._lam = weakform.TrialFunction("lambda_m", mesh, [], mfem.generateCoefficient(0))

    @property
    def trialFunctions(self):
        return [self._m[0],self._m[1],self._m[2], self._lam]

    @property
    def weakform(self):
        a, g = sp.symbols("alpha_G, gam_LL")
        B = sp.Matrix([0,0,sp.Symbol("B3")])

        m = self._m
        test_m = weakform.TestFunction(m)

        lam = self._lam
        test_lam = weakform.TestFunction(lam)

        wf1 =(m.diff(t)/g - a*m.cross(m.diff(t))/g + m.cross(B) + 2*m*lam).dot(test_m) * dV
        wf2 = (m.dot(m) - 1) * dV * test_lam

        return wf1 + wf2

    @property
    def coefficient(self):
        coefs = {"alpha_G": self._mat["LLG"]["alpha"], "gam_LL": mfem.generateCoefficient(1.760859770e11)}
        coefs["B1"] = mfem.generateCoefficient(0)
        coefs["B2"] = mfem.generateCoefficient(0)
        coefs["B3"] = mfem.generateCoefficient(1)
        return coefs