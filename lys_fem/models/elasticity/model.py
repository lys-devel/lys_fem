import numpy as np

from lys_fem import FEMModel, DomainCondition, Coef
from lys_fem.util import grad, dx
from . import InitialCondition, DirichletBoundary


class ThermoelasticStress(DomainCondition):
    className = "ThermoelasticStress"

    def __init__(self, T="T", alpha="alpha", geometries="all", **kwargs):
        super().__init__(geometries = geometries, **kwargs)
        self["T"] = Coef(T, description="Temperature (K)")
        self["alpha"] = Coef(alpha, shape=(3,3), description="Thermal expansion coef. (1/K)")


class DeformationPotential(DomainCondition):
    className = "DeformationPotential"

    def __init__(self, n_e="n_e", n_h="n_h", geometries="all", *args, **kwargs):
        super().__init__(geometries=geometries, *args, **kwargs)
        self["n_e"] = Coef(n_e, description="Electron carrier density (1/m3)")
        self["n_h"] = Coef(n_h, description="Hole carrier density (1/m3)")


class ElasticModel(FEMModel):
    className = "Elasticity"
    boundaryConditionTypes = [DirichletBoundary]
    domainConditionTypes = [ThermoelasticStress, DeformationPotential]
    initialConditionTypes = [InitialCondition]

    def __init__(self, nvar=3, discretization="NewmarkBeta", C="C", rho="rho", *args, **kwargs):
        super().__init__(nvar, *args, varName="u", discretization=discretization, **kwargs)
        self["rho"] = Coef(rho, description="Density (kg/m3)")
        self["C"] = Coef(C, shape=(3,3,3,3), description="Elastic Constant (Pa)")

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