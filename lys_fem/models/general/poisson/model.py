from lys_fem import FEMFixedModel, Equation
from lys_fem.models.common import Source, DivSource, DirichletBoundary, NeumannBoundary


class PoissonEquation(Equation):
    className = "Poisson Equation"
    isScalar = True
    def __init__(self, varName="phi", **kwargs):
        super().__init__(varName, **kwargs)

    def widget(self, fem, canvas):
        from .widgets import PoissonEquationWidget
        return PoissonEquationWidget(self, fem, canvas)
    

class PoissonModel(FEMFixedModel):
    className = "Poisson"
    equationTypes = [PoissonEquation]
    domainConditionTypes = [Source, DivSource]
    boundaryConditionTypes = [DirichletBoundary, NeumannBoundary]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

