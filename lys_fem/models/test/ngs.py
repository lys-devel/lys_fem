from lys_fem.ngs import NGSModel, grad, dx
from . import DirichletBoundary, InitialCondition

class NGSLinearTestModel(NGSModel):
    def __init__(self, model, mesh, vars):
        super().__init__(model, mesh, vars, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            u,v = vars[eq.variableName]
            wf += grad(u).dot(grad(v)) * dx
        return wf


class NGSNonlinearTestModel(NGSModel):
    def __init__(self, model, mesh, vars):
        super().__init__(model, mesh, vars, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            u,v = vars[eq.variableName]
            wf += u.value * grad(u).dot(grad(v)) * dx
        return wf
    

class NGSTwoVariableTestModel(NGSModel):
    def __init__(self, model, mesh, vars):
        super().__init__(model, mesh, vars)
        self._model = model

        init = self._model.initialConditions.coef(InitialCondition)
        dirichlet = self._model.boundaryConditions.coef(DirichletBoundary)
        if dirichlet is None:
            dirichlet = ["auto", "auto"]

        for eq in model.equations:
            self.addVariable("x", 1, dirichlet[0], init[0], region = eq.geometries, order=1)
            self.addVariable("y", 1, dirichlet[1], init[1], region = eq.geometries, order=1)

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            x,test_x = vars["x"]
            y,test_y = vars["y"]
            wf += (x.t*test_x + y.t*test_y) * dx
            wf += (x-y)*test_x *dx + (y-x)*test_y*dx
        return wf


class NGSExpTestModel(NGSModel):
    def __init__(self, model, mesh, vars):
        super().__init__(model, mesh, vars, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            x,test_x = vars["x"]
            wf += (x.t + x)*test_x * dx
        return wf


class NGSTdepFieldTestModel(NGSModel):
    def __init__(self, model, mesh, vars):
        super().__init__(model, mesh, vars, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            y,test_y = vars["y"]
            wf += (y.t + mat["x0"])*test_y * dx
        return wf