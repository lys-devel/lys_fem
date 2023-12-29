from lys_fem import FEMModel, Equation
from . import DirichletBoundary, InitialCondition


class ElasticModel(FEMModel):
    className = "Elasticity"

    def __init__(self, nvar=3, *args, **kwargs):
        super().__init__(nvar, *args, **kwargs)

    @classmethod
    @property
    def equationTypes(self):
        return [ChristffelEquation]

    @classmethod
    @property
    def boundaryConditionTypes(cls):
        return [DirichletBoundary]
    
    @classmethod
    @property
    def initialConditionTypes(cls):
        return [InitialCondition]


class ChristffelEquation(Equation):
    className = "Christffel Equation"
    def __init__(self, varName="u", **kwargs):
        super().__init__(varName, **kwargs)
