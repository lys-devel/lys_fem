from lys_fem import FEMFixedModel, util
from . import DirichletBoundary, DomainCondition, InitialCondition


class RandomForce(DomainCondition):
    className="Random Force"


class LinearTestModel(FEMFixedModel):
    className = "Linear Test"
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="X", isScalar=True, **kwargs)

        
class NonlinearTestModel(FEMFixedModel):
    className = "Nonlinear Test"
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="X", isScalar=True, **kwargs)


class TwoVariableTestModel(FEMFixedModel):
    className = "Two Variable Test"
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, varName="X", **kwargs)

    def functionSpaces(self):
        kwargs = {"size": 1, "isScalar": True, "order": 1, "geometries": self.geometries}
        return [util.FunctionSpace("X", **kwargs), util.FunctionSpace("Y", **kwargs)]
    
    def initialValues(self, params):
        x = super().initialValues(params)[0]
        return [x[0], x[1]]

    def initialVelocities(self, params):
        v = super().initialVelocities(params)[0]
        return [v[0], v[1]]


class ExpTestModel(FEMFixedModel):
    className = "Exp Test"
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="X", isScalar=True, **kwargs)


class TdepFieldTestModel(FEMFixedModel):
    className = "Tdep Field Test"
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="Y", isScalar=True, **kwargs)


class ScaleTestModel(FEMFixedModel):
    className = "Scale Test"
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="X", isScalar=True, **kwargs)


class TwoVarGradTestModel(FEMFixedModel):
    className = "Two Variable Grad Test"
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, varName="X", **kwargs)

    def functionSpaces(self):
        kwargs = {"size": 1, "isScalar": True, "order": 1, "geometries": self.geometries}
        return [util.FunctionSpace("X", **kwargs), util.FunctionSpace("Y", **kwargs)]
    
    def initialValues(self, params):
        x = super().initialValues(params)[0]
        return [x[0], x[1]]

    def initialVelocities(self, params):
        v = super().initialVelocities(params)[0]
        return [v[0], v[1]]


class RandomWalkModel(FEMFixedModel):
    className = "Random Walk Test"
    domainConditionTypes = [RandomForce]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="X", isScalar=True, **kwargs)
