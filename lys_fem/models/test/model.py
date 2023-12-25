from lys_fem import FEMFixedModel, Equation


class LinearTestModel(FEMFixedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(1, [LinearTestEquation("x")], *args, **kwargs)

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


class LinearTestEquation(Equation):
    def __init__(self, varName, domain="all", name="Linear Test Equation"):
        super().__init__(name, varName, geometries = domain)


class NonlinearTestModel(FEMFixedModel):
    def __init__(self, *args, **kwargs):
        super().__init__(1, [NonlinearTestEquation("x")], *args, **kwargs)

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


class NonlinearTestEquation(Equation):
    def __init__(self, varName, domain="all", name="Linear Test Equation"):
        super().__init__(name, varName, geometries = domain)