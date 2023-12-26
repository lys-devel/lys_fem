from lys_fem import FEMFixedModel, Equation


class HeatConductionModel(FEMFixedModel):
    className = "Heat Conduction"

    def __init__(self, *args, **kwargs):
        super().__init__(1, [HeatConductionEquation("T")], *args, **kwargs)

    @classmethod
    @property
    def equationTypes(self):
        return [HeatConductionEquation]
    
    def evalList(self):
        return ["T"]

    def eval(self, data, fem, var):
        if var == "T":
            return data["T"]


class HeatConductionEquation(Equation):
    def __init__(self, varName, domain="all", name="Heat Conduction Equation"):
        super().__init__(name, varName, geometries = domain)
