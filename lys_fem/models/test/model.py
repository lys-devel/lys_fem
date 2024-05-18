from lys_fem import FEMFixedModel, Equation
from . import DirichletBoundary, InitialCondition

class LinearTestEquation(Equation):
    className = "Linear Test Equation"
    isScalar = True
    def __init__(self, varName="x", **kwargs):
        super().__init__(varName, **kwargs)


class NonlinearTestEquation(Equation):
    className = "Nonlinear Test Equation"
    isScalar = True
    def __init__(self, varName="x", **kwargs):
        super().__init__(varName, **kwargs)


class TwoVariableTestEquation(Equation):
    className = "Two Variable Test"
    def __init__(self, varName="x", **kwargs):
        super().__init__(varName, **kwargs)


class LinearTestModel(FEMFixedModel):
    className = "Linear Test"
    equationTypes = [LinearTestEquation]
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

        
class NonlinearTestModel(FEMFixedModel):
    className = "Nonlinear Test"
    equationTypes = [NonlinearTestEquation]
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class TwoVariableTestModel(FEMFixedModel):
    className = "Two Variable Test"
    equationTypes = [TwoVariableTestEquation]
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)

