from lys_fem import FEMFixedModel, Equation


class LinearTestModel(FEMFixedModel):
    className = "Linear Test"

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    @classmethod
    @property
    def equationTypes(cls):
        return [LinearTestEquation]


class LinearTestEquation(Equation):
    className = "Linear Test Equation"
    def __init__(self, varName="x", **kwargs):
        super().__init__(varName, **kwargs)


class NonlinearTestModel(FEMFixedModel):
    className = "Nonlinear Test"

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    @classmethod
    @property
    def equationTypes(cls):
        return [NonlinearTestEquation]


class NonlinearTestEquation(Equation):
    className = "Linear Test Equation"
    def __init__(self, varName="x", **kwargs):
        super().__init__(varName, **kwargs)