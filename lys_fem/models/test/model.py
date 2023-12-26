from lys_fem import FEMFixedModel, Equation


class LinearTestModel(FEMFixedModel):
    className = "Linear Test"

    def __init__(self, *args, **kwargs):
        super().__init__(1, [LinearTestEquation("x")], *args, **kwargs)

    def evalList(self):
        return ["x"]

    def eval(self, data, fem, var):
        if var == "x":
            return data["x"]


class LinearTestEquation(Equation):
    def __init__(self, varName, domain="all", name="Linear Test Equation"):
        super().__init__(name, varName, geometries = domain)


class NonlinearTestModel(FEMFixedModel):
    className = "Nonlinear Test"

    def __init__(self, *args, **kwargs):
        super().__init__(1, [NonlinearTestEquation("x")], *args, **kwargs)

    def evalList(self):
        return ["x"]

    def eval(self, data, fem, var):
        if var == "x":
            return data["x"]


class NonlinearTestEquation(Equation):
    def __init__(self, varName, domain="all", name="Linear Test Equation"):
        super().__init__(name, varName, geometries = domain)