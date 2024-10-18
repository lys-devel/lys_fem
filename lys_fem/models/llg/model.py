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


class MagneticScalarPotential(DomainCondition):
    className = "MagneticScalarPotential"

    @classmethod
    def default(cls, model):
        return MagneticScalarPotential(0)

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Magnetic scalar potential (A)")


class SpinTransferTorque(DomainCondition):
    className = "SpinTransferTorque"

    @classmethod
    def default(cls, model):
        return cls([0,0,0])

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Spin polarized current (A/m^2)")


class LLGModel(FEMFixedModel):
    className = "LLG"
    equationTypes = [LLGEquation]
    domainConditionTypes = [ExternalMagneticField, UniaxialAnisotropy, MagneticScalarPotential, SpinTransferTorque]
    boundaryConditionTypes = [DirichletBoundary]

    def __init__(self, *args, discretization="LLG Asym", constraint="Lagrange", **kwargs):
        super().__init__(3, discretization=discretization, *args, **kwargs)
        self._constraint = constraint

    @property
    def discretizationTypes(self):
        return ["LLG Asym"] + super().discretizationTypes

    @property
    def constraint(self):
        return self._constraint

    @classmethod
    def loadFromDictionary(cls, d):
        m = super().loadFromDictionary(d)
        m._constraint = d["constraint"]
        return m

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["constraint"] = self._constraint
        return d