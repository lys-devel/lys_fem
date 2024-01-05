from lys_fem import FEMFixedModel, Equation, DomainCondition


class LLGModel(FEMFixedModel):
    className = "LLG"
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)

    @classmethod
    @property
    def equationTypes(cls):
        return [LLGEquation]

    @classmethod
    @property
    def domainConditionTypes(cls):
        return [ExternalMagneticField]


class LLGEquation(Equation):
    className = "LLG Equation"
    def __init__(self, varName="m", **kwargs):
        super().__init__(varName, **kwargs)


class GilbertDamping(DomainCondition):
    className = "GilbertDamping"

    @classmethod
    def default(cls, model):
        return cls()
    

class ExternalMagneticField(DomainCondition):
    className = "External Magnetic Field"

    @classmethod
    def default(cls, model):
        return ExternalMagneticField([0,0,0])

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Magnetic Field (T)")


class UniaxialAnisotropy(DomainCondition):
    className = "UniaxialAnisotropy"

    @classmethod
    def default(cls, model):
        return cls()


class Demagnetization(DomainCondition):
    className = "Demagnetization"

    def __init__(self, phi="phi", *args, **kwargs):
        super().__init__(values=phi, *args, **kwargs)

    @classmethod
    def default(cls, model):
        return Demagnetization()

    def widget(self, fem, canvas):
        from lys.Qt import QtWidgets
        return QtWidgets.QWidget()
