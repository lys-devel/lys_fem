from ngsolve.fem import Einsum

from lys_fem.ngs import NGSModel, util
from lys_fem.ngs.util import grad, dx
from . import ThermoelasticStress


class NGSElasticModel(NGSModel):
    def __init__(self, model, mesh, mat):
        super().__init__(model, mesh, addVariables=True, order=2)
        self._model = model
        self._mat = mat
        self._vdim = self._model.variableDimension()

    def weakform(self, vars):
        C, rho = self._mat["C"], self._mat["rho"]
        wf = util.NGSFunction()
        for eq in self._model.equations:
            var = vars[eq.variableName]
            u,v = var.trial, var.test
            gu, gv = grad(u), grad(v)
            
            wf += rho * u.tt.dot(v) * dx
            wf += gu.ddot(C.ddot(gv)) * dx

            t0 = self._model.domainConditions.coef(ThermoelasticStress)
            if t0 is not None:
                alpha = self._mat["alpha"]
                T0 = util.coef(t0, self._mesh, name="T0")
                for te in self._model.domainConditions.get(ThermoelasticStress):
                    T = vars[te.varName].trial
                    wf += (T-T0)*C.ddot(alpha).ddot(gv)*dx
        return wf