from lys_fem.ngs import NGSModel, grad, dx
from . import ThermoelasticStress, DeformationPotential


class NGSElasticModel(NGSModel):
    def __init__(self, model, mesh, vars):
        super().__init__(model, mesh, vars, addVariables=True, order=2)
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
            if self._model.domainConditions.have(DeformationPotential):
                for df in self._model.domainConditions.get(DeformationPotential):
                    d_n, d_p = mat["d_e"], mat["d_h"]
                    n, _ = vars[df.varNames[0]]
                    p, _ = vars[df.varNames[1]]
                    wf += (d_n*n - d_p*p).ddot(gv)*dx
        return wf