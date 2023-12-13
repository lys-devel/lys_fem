from lys_fem import FEMFixedModel


class LinearTestModel(FEMFixedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    @classmethod
    @property
    def name(cls):
        return "Linear Test"

    @property
    def variableName(self):
        return "x"

    def evalList(self):
        return ["x"]

    def eval(self, data, fem, var):
        if var == "x":
            return data["x"]


class NonlinearTestModel(FEMFixedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    @classmethod
    @property
    def name(cls):
        return "Nonlinear Test"

    @property
    def variableName(self):
        return "x"

    def evalList(self):
        return ["x"]

    def eval(self, data, fem, var):
        if var == "x":
            return data["x"]
