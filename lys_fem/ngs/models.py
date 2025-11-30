from . import util, time
from ..models.common import DirichletBoundary


modelList = {}

def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mat):
    return CompositeModel([modelList[m.className](m) for m in fem.models], mat)


class NGSModel:
    def __init__(self, model):
        self._model = model
        self._vars = model.functionSpaces()

    def discretize(self, dti):
        d = {}
        for v in self.variables:
            trial = util.trial(v)
            if self._model.discretization == "ForwardEuler":
                d.update(time.ForwardEuler.generateWeakforms(trial, dti))
            elif self._model.discretization == "BackwardEuler":
                d.update(time.BackwardEuler.generateWeakforms(trial, dti))
            elif self._model.discretization == "BDF2":
                d.update(time.BDF2.generateWeakforms(trial, dti))
            elif self._model.discretization == "NewmarkBeta":
                d.update(time.NewmarkBeta.generateWeakforms(trial, dti))
            else:
                raise RuntimeError("Unknown discretization: "+self._model.discretization)
        return d

    def updater(self, dti):
        d = self.discretize(dti)
        res = {}
        for v in self.variables:
            trial = util.trial(v)
            if trial.t in d:
                res[trial.t] = d[trial.t]
            if trial.tt in d:
                res[trial.tt] = d[trial.tt]
        return res

    @property
    def variables(self):
        return self._vars

    def initialValues(self, mat):
        return self._model.initialValues(mat)

    def initialVelocities(self, mat):
        return self._model.initialVelocities(mat)

    def __str__(self):
        res = "\t"+self._model.className+": discretization = " + self._model.discretization + "\n"
        for i, v in enumerate(self.variables):
            res += "\t\tVariable " + str(1+i) + ": " + str(v)  + "\n"
        return res


class CompositeModel:
    def __init__(self, models, mat):
        self._models = models
        self._mat = mat
        self._mat.update({v.name: util.TrialFunction(v) for v in self.variables})

    def weakforms(self):
        tnt = {v.name: (util.trial(v), util.test(v)) for v in self.variables}
        return sum([model.weakform(tnt, self._mat) for model in self._models])
    
    def discretize(self):
        d = {}
        for m in self._models:
            d.update(m.discretize(util.dti))
        return d

    def updater(self):
        d = {}
        for m in self._models:
            d.update(m.updater(util.dti))
        return d

    def initialValue(self, fes):
        x0 = sum([m.initialValues(self._mat) for m in self._models], [])
        v0 = sum([m.initialVelocities(self._mat) for m in self._models], [])
        x = fes.gridFunction(x0)
        v = fes.gridFunction(v0)
        a = fes.gridFunction()
        return x, v, a
    
    @property
    def materials(self):
        return self._mat

    @property
    def models(self):
        return self._models
    
    @property
    def variables(self):
        return sum([m.variables for m in self._models], [])       