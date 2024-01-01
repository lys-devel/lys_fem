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


class ExternalMagneticField(DomainCondition):
    className = "External Magnetic Field"

    @classmethod
    def default(cls, model):
        return ExternalMagneticField([0,0,0])

    def widget(self, fem, canvas):
        return super().widget(fem, canvas, title="Magnetic Field (T)")
