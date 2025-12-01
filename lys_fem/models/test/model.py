from lys_fem import FEMFixedModel, util
from lys_fem.util import grad, dx
from . import DirichletBoundary, DomainCondition, InitialCondition


class RandomForce(DomainCondition):
    className="Random Force"


class LinearTestModel(FEMFixedModel):
    className = "Linear Test"
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="X", isScalar=True, **kwargs)

    def weakform(self, vars, mat):
        wf = 0
        u,v = vars[self.variableName]
        wf += grad(u).dot(grad(v)) * dx
        return wf


class NonlinearTestModel(FEMFixedModel):
    className = "Nonlinear Test"
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="X", isScalar=True, **kwargs)

    def weakform(self, vars, mat):
        wf = 0
        u,v = vars[self.variableName]
        wf += u * grad(u).dot(grad(v)) * dx
        return wf


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

    def weakform(self, vars, mat):
        x,test_x = vars["X"]
        y,test_y = vars["Y"]

        wf = 0
        wf += (x.t*test_x + y.t*test_y) * dx
        wf += (x-y)*test_x *dx + (y-x)*test_y*dx
        return wf


class ExpTestModel(FEMFixedModel):
    className = "Exp Test"
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="X", isScalar=True, **kwargs)

    def weakform(self, vars, mat):
        wf = 0
        x,test_x = vars["X"]
        wf += (x.t + x)*test_x * dx
        return wf


class TdepFieldTestModel(FEMFixedModel):
    className = "Tdep Field Test"
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="Y", isScalar=True, **kwargs)

    def weakform(self, vars, mat):
        wf = 0
        y,test_y = vars["Y"]
        wf += (y.t + mat["X"])*test_y * dx
        return wf


class ScaleTestModel(FEMFixedModel):
    className = "Scale Test"
    boundaryConditionTypes = [DirichletBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="X", isScalar=True, **kwargs)

    def weakform(self, vars, mat):
        wf = 0
        u,v = vars[self._model.variableName]
        wf += u * grad(u).dot(grad(v)) * dx
        return wf


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

    def weakform(self, vars, mat):
        x,test_x = vars["X"]
        y,test_y = vars["Y"]

        wf = 0
        wf += (x.t*test_x + y.t*test_y) * dx
        wf += x*test_x *dx + grad(x)[0]*test_y*dx
        return wf


class RandomWalkModel(FEMFixedModel):
    className = "Random Walk Test"
    domainConditionTypes = [RandomForce]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="X", isScalar=True, **kwargs)

    def weakform(self, vars, mat):
        wf = 0
        u,v = vars[self.variableName]

        for f in self.domainConditions.get(RandomForce):
            R = mat[f.values]*util.sqrt(util.dti) # Euler-Maruyama formula
            wf += (u.t.dot(v) - R.dot(v)) * dx(f.geometries)
        return wf