import numpy as np

from lys_fem import FEMModel, DomainCondition
from lys_fem.util import grad, dx
from . import InitialCondition, DirichletBoundary


class ThermoelasticStress(DomainCondition):
    className = "ThermoelasticStress"

    def __init__(self, T="T", alpha="alpha", geometries="all", **kwargs):
        super().__init__(geometries = geometries, T=T, alpha=alpha, **kwargs)

    def widget(self, fem, canvas):
        from .widgets import ThermoelasticWidget
        return ThermoelasticWidget(self, fem, canvas, "Temperature T (K)")

    @property
    def description(self):
        return {
            "T": "Temperature (K)",
            "alpha": "Thermal expansion coef. (1/K)",
        }

    @property
    def default(self):
        return {
            "T": 100,
            "alpha": np.eye(3).tolist(),
        }

class DeformationPotential(DomainCondition):
    className = "DeformationPotential"

    def __init__(self, values=["n_e", "n_h"], *args, **kwargs):
        super().__init__(n_e=values[0], n_h=values[1], *args, **kwargs)

    def widget(self, fem, canvas):
        from .widgets import DeformationPotentialWidget
        return DeformationPotentialWidget(self, fem, canvas)


class ElasticModel(FEMModel):
    className = "Elasticity"
    boundaryConditionTypes = [DirichletBoundary]
    domainConditionTypes = [ThermoelasticStress, DeformationPotential]
    initialConditionTypes = [InitialCondition]

    def __init__(self, nvar=3, discretization="NewmarkBeta", C="C", rho="rho", *args, **kwargs):
        super().__init__(nvar, *args, varName="u", C=C, rho=rho, discretization=discretization, **kwargs)

    def weakform(self, vars, mat):
        C, rho = mat[self.C], mat[self.rho]
        wf = 0

        u,v = vars[self.variableName]
        gu, gv = grad(u), grad(v)
        
        wf += rho * u.tt.dot(v) * dx
        wf += gu.ddot(C.ddot(gv)) * dx

        for te in self.domainConditions.get(ThermoelasticStress):
            alpha, T = mat[te.alpha], mat[te.T]
            wf += T*C.ddot(alpha).ddot(gv)*dx(te.geometries)

        for df in self.domainConditions.get(DeformationPotential):
            d_n, d_p = mat["-d_e*e"], mat["-d_h*e"]
            n,p = mat[df.n_e], mat[df.n_h]
            I = mat[np.eye(3)]
            wf += (d_n*n - d_p*p)*gv.ddot(I)*dx(df.geometries)
        return wf