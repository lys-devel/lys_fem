import numpy as np
from lys_fem import FEMFixedModel, util
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
