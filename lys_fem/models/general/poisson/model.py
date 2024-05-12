from lys_fem import FEMFixedModel, Equation
from lys_fem.models.common import Source, DirichletBoundary, NeumannBoundary


class PoissonEquation(Equation):
    className = "Poisson Equation"
    isScalar = True
    def __init__(self, varName="phi", **kwargs):
        super().__init__(varName, **kwargs)


class PoissonModel(FEMFixedModel):
    className = "Poisson"
    equationTypes = [PoissonEquation]
    domainConditionTypes = [Source]
    boundaryConditionTypes = [DirichletBoundary, NeumannBoundary]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)




