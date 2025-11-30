from lys_fem.ngs import NGSModel, grad, dx, util
from . import DirichletBoundary, InitialCondition, RandomForce

class NGSLinearTestModel(NGSModel):
    def __init__(self, model):
        super().__init__(model)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        u,v = vars[self._model.variableName]
        wf += grad(u).dot(grad(v)) * dx
        return wf


class NGSNonlinearTestModel(NGSModel):
    def __init__(self, model):
        super().__init__(model)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        u,v = vars[self._model.variableName]
        wf += u * grad(u).dot(grad(v)) * dx
        return wf
    

class NGSTwoVariableTestModel(NGSModel):
    def __init__(self, model):
        super().__init__(model)
        self._model = model

    def weakform(self, vars, mat):
        x,test_x = vars["X"]
        y,test_y = vars["Y"]

        wf = 0
        wf += (x.t*test_x + y.t*test_y) * dx
        wf += (x-y)*test_x *dx + (y-x)*test_y*dx
        return wf


class NGSExpTestModel(NGSModel):
    def __init__(self, model):
        super().__init__(model)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        x,test_x = vars["X"]
        wf += (x.t + x)*test_x * dx
        return wf


class NGSTdepFieldTestModel(NGSModel):
    def __init__(self, model):
        super().__init__(model)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        y,test_y = vars["Y"]
        wf += (y.t + mat["X"])*test_y * dx
        return wf
    

class NGSScaleTestModel(NGSModel):
    def __init__(self, model):
        super().__init__(model)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        u,v = vars[self._model.variableName]
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
        
    def weakform(self, vars, mat):
        x,test_x = vars["X"]
        y,test_y = vars["Y"]

        wf = 0
        wf += (x.t*test_x + y.t*test_y) * dx
        wf += x*test_x *dx + grad(x)[0]*test_y*dx
        return wf
    

class NGSRandomWalkModel(NGSModel):
    def __init__(self, model):
        super().__init__(model)
        self._model = model

    def weakform(self, vars, mat):
        wf = 0
        u,v = vars[self._model.variableName]

        for f in self._model.domainConditions.get(RandomForce):
            R = mat[f.values]*util.sqrt(util.dti) # Euler-Maruyama formula
            wf += (u.t.dot(v) - R.dot(v)) * dx(f.geometries)
        return wf