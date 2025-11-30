

from lys_fem import FEMFixedModel, DomainCondition, util
from . import DirichletBoundary


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


class BarnettEffect(DomainCondition):
    className = "BarnettEffect"

    @classmethod
    def default(cls, fem, model):
        return BarnettEffect("u")

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
    domainConditionTypes = [ExternalMagneticField, UniaxialAnisotropy, CubicAnisotropy, MagneticScalarPotential, CubicMagnetoStriction, CubicMagnetoRotationCoupling, BarnettEffect, SpinTransferTorque, ThermalFluctuation]
    boundaryConditionTypes = [DirichletBoundary]

    def __init__(self, *args, constraint="Lagrange", order=2, **kwargs):
        valtype = "v" if constraint=="Alouges" else "x"
        super().__init__(3, *args, order=order, varName="m", valType=valtype, **kwargs)
        self._constraint = constraint

    @property
    def discretizationTypes(self):
        return ["LLG Asym"] + super().discretizationTypes

    @property
    def constraint(self):
        return self._constraint

    def functionSpaces(self):
        fes = super().functionSpaces()[0]

        kwargs = {"size": 1, "isScalar": True}
        if self.constraint == "Alouges":
            kwargs.update({"order": self.order-1, "fetype": self.fetype})
        elif self.constraint == "Lagrange":
            kwargs.update({"order": 0, "fetype": "L2"})
        else:
            return [fes]

        return [fes, util.FunctionSpace(self.variableName+"_lam", geometries=self.geometries, **kwargs)]

    def initialValues(self, params):
        x0 = super().initialValues(params)
        if self.constraint in ["Alouges", "Lagrange"]:
            x0.append(util.eval(0))
        return x0

    def initialVelocities(self, params):
        v0 = super().initialVelocities(params)
        if self.constraint in ["Alouges", "Lagrange"]:
            v0.append(util.eval(0))
        return v0

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

