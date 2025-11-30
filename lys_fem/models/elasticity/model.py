from lys_fem import FEMModel, DomainCondition
from . import InitialCondition, DirichletBoundary


class ThermoelasticStress(DomainCondition):
    className = "ThermoelasticStress"

    def __init__(self, values="T", *args, **kwargs):
        super().__init__(values=values, *args, **kwargs)

    def widget(self, fem, canvas):
        from .widgets import ThermoelasticWidget
        return ThermoelasticWidget(self, fem, canvas, "Temperature T (K)")


class DeformationPotential(DomainCondition):
    className = "DeformationPotential"

    def __init__(self, values=["n_e", "n_h"], *args, **kwargs):
        super().__init__(values=values, *args, **kwargs)

    def widget(self, fem, canvas):
        from .widgets import DeformationPotentialWidget
        return DeformationPotentialWidget(self, fem, canvas)


class ElasticModel(FEMModel):
    className = "Elasticity"
    boundaryConditionTypes = [DirichletBoundary]
    domainConditionTypes = [ThermoelasticStress, DeformationPotential]
    initialConditionTypes = [InitialCondition]

    def __init__(self, nvar=3, discretization="NewmarkBeta", *args, **kwargs):
        super().__init__(nvar, *args, varName="u", discretization=discretization, **kwargs)


