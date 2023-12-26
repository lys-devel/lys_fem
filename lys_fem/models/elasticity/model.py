from lys_fem import FEMModel, Equation


class ElasticModel(FEMModel):
    def __init__(self, nvar=3, *args, **kwargs):
        super().__init__(nvar, [ChristffelEquation("u")], *args, **kwargs)

    @classmethod
    @property
    def name(cls):
        return "Elasticity"

    @classmethod
    @property
    def equationTypes(self):
        return [ChristffelEquation]

    def evalList(self):
        return ["u", "ux", "uy", "uz"]

    def eval(self, data, fem, var):
        if var == "u":
            return data["u"]
        if var == "ux":
            return data["u1"]
        if var == "uy":
            return data["u2"]
        if var == "uz":
            return data["u3"]



class ChristffelEquation(Equation):
    def __init__(self, varName, domain="all", name="Christffel Equation"):
        super().__init__(name, varName, geometries=domain)
