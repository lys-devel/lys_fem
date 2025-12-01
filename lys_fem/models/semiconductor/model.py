import numpy as np
from lys_fem import FEMFixedModel, util
from lys_fem.util import grad, dx
from .. import common
from . import DirichletBoundary


class InitialCondition(common.InitialCondition):
    className="Initial Condition"

    @staticmethod
    def fromDensities(ni, Nd=0, Na=0, *args, **kwargs):
        dN = (Nd-Na)/2
        n0 = dN + np.sqrt(dN**2 + ni**2)
        p0 = -dN + np.sqrt(dN**2 + ni**2)
        return InitialCondition([n0, p0], *args, **kwargs)

    @classmethod
    def default(cls, model):
        return InitialCondition([1e16,1e16])

    def widget(self, fem, canvas, title="Initial Value"):
        return super().widget(fem, canvas, title)


class SemiconductorModel(FEMFixedModel):
    className = "Semiconductor"
    initialConditionTypes = [InitialCondition]
    boundaryConditionTypes = [DirichletBoundary]

    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, varName="n", **kwargs)
        self.phi = "phi"
        self.T = None

    def functionSpaces(self):
        dirichlet = self.boundaryConditions.dirichlet
        if dirichlet is None:
            dirichlet = [None, None]
        kwargs = {"size": 1, "isScalar": True, "order": self.order, "geometries": self.geometries}
        return [util.FunctionSpace(self.variableName+"_e", dirichlet=[dirichlet[0]], **kwargs), util.FunctionSpace(self.variableName+"_h", dirichlet=[dirichlet[1]], **kwargs)]
    
    def initialValues(self, params):
        x = super().initialValues(params)[0]
        return [x[0], x[1]]

    def initialVelocities(self, params):
        v = super().initialVelocities(params)[0]
        return [v[0], v[1]]

    def weakform(self, vars, mat):
        q, kB = 1.602176634e-19, 1.3806488e-23
        mu_n, mu_p, Nd, Na = mat["mu_n"], mat["mu_p"] , mat["N_d"], mat["N_a"]

        wf = 0
        n, test_n = vars[self.variableName+"_e"]
        p, test_p = vars[self.variableName+"_h"]
        phi, test_phi = vars[self.phi]
        if self.T is None:
            T = mat["T"]
        else:
            T = vars[self.T][0]
        D_n, D_p = mu_n*kB*T/q, mu_p*kB*T/q

        # lhs, drift current, diffusion current terms
        wf += (n.t.dot(test_n) + p.t.dot(test_p))*dx
        wf += grad(phi).dot(-n*mu_n*grad(test_n) + p*mu_p*grad(test_p))*dx
        wf += D_n*grad(n).dot(grad(test_n))*dx + D_p*grad(p).dot(grad(test_p))*dx

        wf -= q*(p-n+Nd-Na)*test_phi * dx 

        return wf
