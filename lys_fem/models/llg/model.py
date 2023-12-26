from lys_fem import FEMFixedModel, Equation


class LLGModel(FEMFixedModel):
    className = "LLG"

    def __init__(self, *args, **kwargs):
        super().__init__(3, [LLGEquation("m")], *args, **kwargs)

    def evalList(self):
        return ["mx", "my", "mz"]

    def eval(self, data, fem, var):
        if var == "mx":
            return data["m1"]
        if var == "my":
            return data["m2"]
        if var == "mz":
            return data["m3"]


class LLGEquation(Equation):
    def __init__(self, varName, domain="all", name="LLG Equation"):
        super().__init__(name, varName, geometries = domain)
