import numpy as np
from lys_fem import FEMFixedModel, Equation, DomainCondition
from .. import common
from . import DirichletBoundary

class SemiconductorDriftDiffusionEquation(Equation):
    className = "Semiconductor Drift Diffusion Equation"
    def __init__(self, varName="n", potential="phi", temperature=None, **kwargs):
        super().__init__(varName, **kwargs)
        self._potName=potential
        self._temp = temperature

    @property
    def potName(self):
        return self._potName
    
    @property
    def tempName(self):
        return self._temp


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
    equationTypes = [SemiconductorDriftDiffusionEquation]
    initialConditionTypes = [InitialCondition]
    boundaryConditionTypes = [DirichletBoundary]

    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)