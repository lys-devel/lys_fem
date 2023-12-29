from lys_fem import FEMFixedModel, Equation


class LLGModel(FEMFixedModel):
    className = "LLG"

    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)

    @classmethod
    @property
    def equationTypes(cls):
        return [LLGEquation]

class LLGEquation(Equation):
    className = "LLG Equation"
    def __init__(self, varName="m", **kwargs):
        super().__init__(varName, **kwargs)
