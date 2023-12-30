from lys_fem import FEMFixedModel, Equation
from lys_fem.models.common import Source, DirichletBoundary, NeumannBoundary

class PoissonModel(FEMFixedModel):
    className = "Poisson"

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    @classmethod
    @property
    def equationTypes(cls):
        return [PoissonEquation]

    @classmethod
    @property
    def domainConditionTypes(cls):
        return [Source]
    
    @classmethod
    @property
    def boundaryConditionTypes(cls):
        return [DirichletBoundary, NeumannBoundary]


class PoissonEquation(Equation):
    className = "Poisson Equation"
    def __init__(self, varName="phi", **kwargs):
        super().__init__(varName, **kwargs)



