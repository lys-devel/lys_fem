from lys_fem.ngs import NGSModel, grad, dx, util
from . import DirichletBoundary, InitialCondition, RandomForce

class NGSLinearTestModel(NGSModel):
    def __init__(self, model):
        super().__init__(model, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            u,v = vars[eq.variableName]
            wf += grad(u).dot(grad(v)) * dx
        return wf


class NGSNonlinearTestModel(NGSModel):
    def __init__(self, model):
        super().__init__(model, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            u,v = vars[eq.variableName]
            wf += u * grad(u).dot(grad(v)) * dx
        return wf
    

class NGSTwoVariableTestModel(NGSModel):
    def __init__(self, model):
        super().__init__(model)
        self._model = model

        init = self._model.initialConditions.coef(InitialCondition)
        dirichlet = self._model.boundaryConditions.coef(DirichletBoundary)
        if dirichlet is None:
            dirichlet = ["auto", "auto"]

        for eq in model.equations:
            self.addVariable("X", 1, self._pick(dirichlet, 0), self._pick(init, 0), region = eq.geometries, order=1, isScalar=True)
            self.addVariable("Y", 1, self._pick(dirichlet, 1), self._pick(init, 1), region = eq.geometries, order=1, isScalar=True)

    def _pick(self, coef, index):
        if isinstance(coef, dict):
            return {key: value[index] for key, value in coef.items()}
        else:
            return coef[index]

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            x,test_x = vars["X"]
            y,test_y = vars["Y"]
            wf += (x.t*test_x + y.t*test_y) * dx
            wf += (x-y)*test_x *dx + (y-x)*test_y*dx
        return wf


class NGSExpTestModel(NGSModel):
    def __init__(self, model):
        super().__init__(model, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            x,test_x = vars["X"]
            wf += (x.t + x)*test_x * dx
        return wf


class NGSTdepFieldTestModel(NGSModel):
    def __init__(self, model):
        super().__init__(model, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            y,test_y = vars["Y"]
            wf += (y.t + mat["X"])*test_y * dx
        return wf
    

class NGSScaleTestModel(NGSModel):
    def __init__(self, model):
        super().__init__(model, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            u,v = vars[eq.variableName]
            wf += u * grad(u).dot(grad(v)) * dx
        return wf
    
    @property
    def scale(self):
        return 10
    
    @property
    def residualScale(self):
        return 2


class NGSTwoVarGradTestModel(NGSModel):
    def __init__(self, model):
        super().__init__(model)
        self._model = model

        init = self._model.initialConditions.coef(InitialCondition)
        dirichlet = self._model.boundaryConditions.coef(DirichletBoundary)
        if dirichlet is None:
            dirichlet = ["auto", "auto"]

        for eq in model.equations:
            self.addVariable("X", 1, self._pick(dirichlet, 0), self._pick(init, 0), region = eq.geometries, order=1, isScalar=True)
            self.addVariable("Y", 1, self._pick(dirichlet, 1), self._pick(init, 1), region = eq.geometries, order=1, isScalar=True)

    def _pick(self, coef, index):
        if isinstance(coef, dict):
            return {key: value[index] for key, value in coef.items()}
        else:
            return coef[index]
        
    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            x,test_x = vars["X"]
            y,test_y = vars["Y"]
            wf += (x.t*test_x + y.t*test_y) * dx
            wf += x*test_x *dx + grad(x)[0]*test_y*dx
        return wf
    

class NGSRandomWalkModel(NGSModel):
    def __init__(self, model):
        super().__init__(model, addVariables=True)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        for eq in self._model.equations:
            u,v = vars[eq.variableName]

            for f in self._model.domainConditions.get(RandomForce):
                R = mat[f.values]*util.sqrt(util.dti) # Euler-Maruyama formula
                wf += (u.t.dot(v) - R.dot(v)) * dx(f.geometries)
        return wf