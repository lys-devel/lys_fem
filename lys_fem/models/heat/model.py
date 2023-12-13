from lys_fem import FEMFixedModel


class HeatConductionModel(FEMFixedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    @classmethod
    @property
    def name(cls):
        return "Heat Conduction"

    @property
    def variableName(self):
        return "T"

    def evalList(self):
        return ["T"]

    def eval(self, data, fem, var):
        if var == "T":
            return data["T"]
