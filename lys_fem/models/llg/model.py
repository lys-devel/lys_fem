from lys_fem import FEMFixedModel


class LLGModel(FEMFixedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)

    @classmethod
    @property
    def name(cls):
        return "LLG"

    @property
    def variableName(self):
        return "m"

    def evalList(self):
        return ["mx", "my", "mz"]

    def eval(self, data, fem, var):
        if var == "mx":
            return data["m1"][:, 0]
        if var == "my":
            return data["m2"][:, 0]
        if var == "mz":
            return data["m3"][:, 0]
