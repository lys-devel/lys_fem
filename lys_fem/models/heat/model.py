from lys_fem import FEMFixedModel
from . import DirichletBoundary, NeumannBoundary, InitialCondition


class HeatConductionModel(FEMFixedModel):
    className = "Heat Conduction"
    boundaryConditionTypes = [DirichletBoundary, NeumannBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="T", isScalar=True, **kwargs)
