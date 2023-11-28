from lys_fem import FEMModel


class ElasticModel(FEMModel):
    def __init__(self, nvar=3, *args, **kwargs):
        super().__init__(nvar, *args, **kwargs)

    @classmethod
    @property
    def name(cls):
        return "Elasticity"

    @property
    def variableName(self):
        return "u"

    def evalList(self):
        return ["ux", "uy", "uz"]

    def eval(self, data, fem, var):
        if var == "ux":
            return data["u1"][:, 0]
        if var == "uy":
            return data["u2"][:, 0]
        if var == "uz":
            return data["u3"][:, 0]
