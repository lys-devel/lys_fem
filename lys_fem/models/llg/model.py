

from lys_fem import FEMFixedModel, Equation, DomainCondition, GeometrySelection
from . import DirichletBoundary

class LLGEquation(Equation):
    className = "LLG Equation"
    def __init__(self, varName="m", **kwargs):
        super().__init__(varName, **kwargs)


class ExternalMagneticField(DomainCondition):
    className = "External Magnetic Field"

    @classmethod
    def default(cls, fem, model):
        return ExternalMagneticField([0,0,0])

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Magnetic Field (T)")


class UniaxialAnisotropy(DomainCondition):
    className = "UniaxialAnisotropy"


class CubicAnisotropy(DomainCondition):
    className = "CubicAnisotropy"


class CubicMagnetoStriction(DomainCondition):
    className = "CubicMagnetoStriction"

    @classmethod
    def default(cls, fem, model):
        return CubicMagnetoStriction("u")

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Displacement field (m)")


class CubicMagnetoRotationCoupling(DomainCondition):
    className = "CubicMagnetoRotationCoupling"

    @classmethod
    def default(cls, fem, model):
        return CubicMagnetoRotationCoupling("u")

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Displacement field (m)")


class MagneticScalarPotential(DomainCondition):
    className = "MagneticScalarPotential"

    @classmethod
    def default(cls, fem, model):
        return MagneticScalarPotential(0)

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Magnetic scalar potential (A)")


class SpinTransferTorque(DomainCondition):
    className = "SpinTransferTorque"

    @classmethod
    def default(cls, fem, model):
        return cls([0]*fem.dimension)

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Spin polarized current (A/m^2)")


class ThermalFluctuation(DomainCondition):
    className = "ThermalFluctuation"

    @classmethod
    def default(cls, fem, model):
        return cls(T=0, R=[0,0,0])

    def widget(self, fem, canvas):
        from .widgets import ThermalFluctuationWidget
        return ThermalFluctuationWidget(self)


class LLGModel(FEMFixedModel):
    className = "LLG"
    equationTypes = [LLGEquation]
    domainConditionTypes = [ExternalMagneticField, UniaxialAnisotropy, CubicAnisotropy, MagneticScalarPotential, CubicMagnetoStriction, CubicMagnetoRotationCoupling, SpinTransferTorque, ThermalFluctuation]
    boundaryConditionTypes = [DirichletBoundary]

    def __init__(self, *args, constraint="Lagrange", order=2, **kwargs):
        super().__init__(3, order=order, *args, **kwargs)
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
        if "constraint" in d:
            m._constraint = d["constraint"]
        return m

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["constraint"] = self._constraint
        return d

    def widget(self, fem, canvas):
        from .widgets import LLGModelWidget
        return LLGModelWidget(self)

