from lys_fem import FEMFixedModel, Equation
from .. import common


class HeatConductionEquation(Equation):
    className = "Heat Conduction Equation"
    def __init__(self, varName="T", **kwargs):
        super().__init__(varName, **kwargs)


class NeumannBoundary(common.NeumannBoundary):
    unit = "W/m^2"


class InitialCondition(common.InitialCondition):
    unit = "K"
    

class HeatConductionModel(FEMFixedModel):
    className = "Heat Conduction"
    equationTypes = [HeatConductionEquation]
    boundaryConditionTypes = [NeumannBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)




