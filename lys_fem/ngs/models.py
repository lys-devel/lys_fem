from lys_fem.fem import FEMCoefficient
from . import util, time
from ..models.common import DirichletBoundary


modelList = {}

def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mat):
    return CompositeModel([modelList[m.className](m) for m in fem.models], mat)


class InitialConditions:
    def __init__(self, space, x, v):
        self._sp = space
        self._x = x
        self._v = v

    def value(self, vars):
        coef = vars[self._x]
        if coef is None:
            raise RuntimeError("Invalid initial value for " + str(self._sp.name))
        if not coef.valid:
            coef = util.NGSFunction([0] * self._sp.size)
        return coef
    
    def velocity(self, vars):
        coef = vars[self._v]
        if coef is None:
            raise RuntimeError("Invalid initial velocity for " + str(self._sp.name))
        if not coef.valid:
            coef = util.NGSFunction([0] * self._sp.size)
        return coef


class NGSModel:
    def __init__(self, model, addVariables=False):
        self._model = model
        self._vars = []
        self._inits = []

        if addVariables:
            for eq in model.equations:
                self.addVariable(eq.variableName, eq.variableDimension, region=eq.geometries, order=model.order, isScalar=eq.isScalar, fetype=model.type)

    def addVariable(self, name, vdim, dirichlet="auto", initialValue="auto", initialVelocity=None, region=None, order=1, isScalar=False, type="x", fetype="H1"):
        kwargs = {"fetype": fetype, "isScalar": isScalar, "order": order, "size": vdim, "valtype": type}
        if region is not None:
            if region.selectionType() == "Selected":
                kwargs["definedon"] = "|".join([region.geometryType.lower() + str(r) for r in region])

        if dirichlet == "auto":
            dirichlet = self._model.boundaryConditions.coef(DirichletBoundary)
        dirichlet = self.__dirichlet(dirichlet, vdim)

        if dirichlet is not None:
            kwargs["dirichlet"] = ["|".join(["boundary" + str(item) for item in dirichlet[i]]) for i in range(vdim)]

        fes = util.FunctionSpace(name, **kwargs)
        self._vars.append(fes)
        self._inits.append(InitialConditions(fes, self.__initialValue(vdim, initialValue), self.__initialVelocity(vdim, initialVelocity)))

    def __dirichlet(self, coef, vdim):
        bdr_dir = [[] for _ in range(vdim)] 
        if coef is None:
            return bdr_dir
        for key, value in coef.value.items():
            for i, bdr in enumerate(bdr_dir):
                if hasattr(value, "__iter__"):
                    if value[i]:
                        bdr.append(key)
                elif value:
                    bdr.append(key)
        return list(bdr_dir)

    def __initialValue(self, vdim, initialValue):
        if initialValue is None:
            return FEMCoefficient([0]*vdim)
        if initialValue != "auto":
            return initialValue
        init = self._model.initialConditions.coef(self._model.initialConditionTypes[0])
        for type in self._model.initialConditionTypes[1:]:
            init.value.update(self._model.initialConditions.coef(type).value)
        return init

    def __initialVelocity(self, vdim, initialVelocity):
        if initialVelocity is None:
            initialVelocity = FEMCoefficient([0]*vdim)
        return initialVelocity

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

    @property
    def initialConditions(self):
        return self._inits

    @property
    def name(self):
        return self._model.name
    
    @property
    def discretization(self):
        return self._model.discretization
    
    @property
    def scale(self):
        return 1
    
    @property
    def residualScale(self):
        return 1


class CompositeModel:
    def __init__(self, models, mat):
        self._models = models
        self._mat = mat
        self._mat.update({v.name: util.TrialFunction(v) for v in self.variables})

    def weakforms(self, type="raw", sols=None, symbols=None):
        tnt = {v.name: (util.trial(v), util.test(v)) for v in self.variables}
        wf = sum([model.weakform(tnt, self._mat) for model in self._models])
        if type == "discretized":
            wf = self.__discretizeWeakform(wf, symbols)
        if type == "initial":
            wf = self.__initial(wf)
        if sols is not None:
            wf = wf.replace(sols.prevDict)
        return wf

    def __discretizeWeakform(self, wf, symbols):
        d = dict(self.discretize())
        if symbols is not None:
            for v in self.variables:
                trial, test = util.trial(v), util.test(v)
                if v.name not in symbols:
                    d.update({trial: util.prev(trial), trial.t: util.prev(trial.t), trial.tt: util.prev(trial.tt), test:0, util.grad(test): 0})
        return wf.replace(d)

    def __initial(self, wf):
        d = {}
        for v in self.variables:
            trial, test = util.trial(v), util.test(v)
            if trial.tt in wf:
                d.update({trial: util.prev(trial), trial.t: util.prev(trial.t), trial.tt: trial, util.grad(trial): util.grad(util.prev(trial))})
            else:
                d.update({trial: util.prev(trial), trial.t: util.prev(trial.t), util.grad(trial): util.grad(util.prev(trial.t)), test: 0, util.grad(test): 0})
        return wf.replace(d)
    
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
        inits = sum([m.initialConditions for m in self._models], [])
        x = fes.gridFunction([v.value(self._mat) for v in inits])
        v = fes.gridFunction([v.velocity(self._mat) for v in inits])
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
