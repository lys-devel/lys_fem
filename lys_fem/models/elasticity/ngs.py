from lys_fem.ngs import NGSModel, grad, dx
from . import ThermoelasticStress


class NGSElasticModel(NGSModel):
    def __init__(self, model, mesh):
        super().__init__(model, mesh, addVariables=True, order=2)
        self._model = model

    def weakform(self, vars, mat):
        C, rho = mat["C"], mat["rho"]
        wf = 0
        for eq in self._model.equations:
            u,v = vars[eq.variableName]
            gu, gv = grad(u), grad(v)
            
            wf += rho * u.tt.dot(v) * dx
            wf += gu.ddot(C.ddot(gv)) * dx

            if self._model.domainConditions.have(ThermoelasticStress):
                alpha = mat["alpha"]
                T0 = self.coef(ThermoelasticStress, name="T0")
                for te in self._model.domainConditions.get(ThermoelasticStress):
                    T, _ = vars[te.varName]
                    wf += (T-T0)*C.ddot(alpha).ddot(gv)*dx
        return wf