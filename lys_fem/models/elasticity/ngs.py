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

            print(C.eval().shape, gv.eval().shape)
            ex = rho*u.tt.dot(v)*dx
            ex2 =  C.ddot(gv)
            print(type(ex2.eval()))

            t0 = self._model.domainConditions.coef(ThermoelasticStress)
            if t0 is not None:
                alpha = self._mat["alpha"]
                T0 = util.generateCoefficient(t0, self._mesh)
                for te in self._model.domainConditions.get(ThermoelasticStress):
                    T, test_T = tnt[te.varName]
                    if self._vdim == 1:
                        K += T*C*alpha*gv*dx
                        F += T0*C*alpha*gv*dx
                    else:
                        beta = Einsum("ijkl,kl->ij", C, alpha)
                        K += T*Einsum("ij,ij", beta, gv)*dx
                        F += T0*Einsum("ij,ij", beta, gv)*dx
        return wf