from lys_fem import FEMFixedModel, Equation
from . import DirichletBoundary, NeumannBoundary, InitialCondition


class HeatConductionEquation(Equation):
    className = "Heat Conduction Equation"
    isScalar = True
    def __init__(self, varName="T", **kwargs):
        super().__init__(varName, **kwargs)


class HeatConductionModel(FEMFixedModel):
    className = "Heat Conduction"
    equationTypes = [HeatConductionEquation]
    boundaryConditionTypes = [DirichletBoundary, NeumannBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)
