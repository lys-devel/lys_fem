from lys_fem import FEMFixedModel, Equation


class LLGModel(FEMFixedModel):
    className = "LLG"

    def __init__(self, *args, **kwargs):
        super().__init__(3, [LLGEquation("m")], *args, **kwargs)


class LLGEquation(Equation):
    def __init__(self, varName, domain="all", name="LLG Equation"):
        super().__init__(name, varName, geometries = domain)
