from lys_fem import FEMFixedModel, Equation, BoundaryCondition
from .. import common

class HeatConductionModel(FEMFixedModel):
    className = "Heat Conduction"

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    @classmethod
    @property
    def equationTypes(self):
        return [HeatConductionEquation]

    @classmethod
    @property
    def boundaryConditionTypes(self):
        return [NeumannBoundary]

    @classmethod
    @property
    def initialConditionTypes(self):
        return [InitialCondition]


class HeatConductionEquation(Equation):
    className = "Heat Conduction Equation"
    def __init__(self, varName="T", **kwargs):
        super().__init__(varName, **kwargs)


class NeumannBoundary(common.NeumannBoundary):
    @classmethod
    @property
    def unit(cls):
        return "W/m^2"


class InitialCondition(common.InitialCondition):
    @classmethod
    @property
    def unit(cls):
        return "K"