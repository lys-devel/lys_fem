import numpy as np
from lys_fem import FEMFixedModel, Coef
from lys_fem.util import dx, grad, det
from lys_fem.models.common import Source, DivSource, DirichletBoundary, NeumannBoundary


class PoissonModel(FEMFixedModel):
    className = "Poisson"
    domainConditionTypes = [Source, DivSource]
    boundaryConditionTypes = [DirichletBoundary, NeumannBoundary]

    def __init__(self, *args, J=None, k=None, **kwargs):
        super().__init__(1, *args, varName="phi", isScalar=True, **kwargs)
        self["J"] = Coef(J, description="Jacobian matrix", shape=(3,3), default="J")
        self["k"] = Coef(k, description="Diffusion constant", shape=(3,3), default=np.eye(3))
    
    def weakform(self, vars, mat):
        wf = 0
        u, v = vars[self.variableName]

        J = mat[self.J]
        if J is not None:
            gu = J.dot(grad(u))
            gv = J.dot(grad(v))
            d = det(J)
        else:
            gu = grad(u)
            gv = grad(v)
            d = 1

        k = mat[self.k]
        if k is not None:
            gu = k.dot(gu)
        wf += gu.dot(gv)/d * dx

        for s in self.domainConditions.get(Source):
            f = mat[s.value]
            wf += f*v*dx(s.geometries)
        
        for s in self.domainConditions.get(DivSource):
            f = mat[s.value]
            wf += -f.dot(grad(v))*dx(s.geometries)
        return wf

class AxialPoissonModel(FEMFixedModel):
    className = "AxialPoisson"
    boundaryConditionTypes = [DirichletBoundary, NeumannBoundary]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="phi", isScalar=True, **kwargs)
    
    def weakform(self, vars, mat):
        wf = 0
        u, v = vars[self.variableName]

        gu = grad(u)
        gv = grad(v)
        wf += gu.dot(gv) * mat["x"] * dx
        return wf
