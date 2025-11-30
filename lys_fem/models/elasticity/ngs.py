import numpy as np
from lys_fem.ngs import NGSModel, grad, dx
from . import ThermoelasticStress, DeformationPotential


class NGSElasticModel(NGSModel):
    def __init__(self, model):
        super().__init__(model)
        self._model = model

    def weakform(self, vars, mat):
        C, rho = mat["C"], mat["rho"]
        wf = 0

        u,v = vars[self._model.variableName]
        gu, gv = grad(u), grad(v)
        
        wf += rho * u.tt.dot(v) * dx
        wf += gu.ddot(C.ddot(gv)) * dx

        for te in self._model.domainConditions.get(ThermoelasticStress):
            alpha, T = mat["alpha"], mat[te.values]
            wf += T*C.ddot(alpha).ddot(gv)*dx(te.geometries)

        for df in self._model.domainConditions.get(DeformationPotential):
            d_n, d_p = mat["-d_e*e"], mat["-d_h*e"]
            n,p = mat[df.values[0]], mat[df.values[1]]
            I = mat[np.eye(3)]
            wf += (d_n*n - d_p*p)*gv.ddot(I)*dx(df.geometries)
        return wf