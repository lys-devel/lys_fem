from lys_fem import FEMFixedModel, Equation, DomainCondition
from . import DirichletBoundary

class LLGEquation(Equation):
    className = "LLG Equation"
    def __init__(self, varName="m", **kwargs):
        super().__init__(varName, **kwargs)


class ExternalMagneticField(DomainCondition):
    className = "External Magnetic Field"

    @classmethod
    def default(cls, model):
        return ExternalMagneticField([0,0,0])

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Magnetic Field (T)")


class UniaxialAnisotropy(DomainCondition):
    className = "UniaxialAnisotropy"


class GilbertDamping(DomainCondition):
    className = "GilbertDamping"


class Demagnetization(DomainCondition):
    className = "Demagnetization"

    def __init__(self, values="phi", *args, **kwargs):
        super().__init__(values=values, *args, **kwargs)

    def widget(self, fem, canvas):
        from lys.Qt import QtWidgets
        return QtWidgets.QWidget()


class LLGModel(FEMFixedModel):
    className = "LLG"
    equationTypes = [LLGEquation]
    domainConditionTypes = [ExternalMagneticField, UniaxialAnisotropy, GilbertDamping, Demagnetization]
    boundaryConditionTypes = [DirichletBoundary]

    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)


