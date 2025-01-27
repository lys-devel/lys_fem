from lys_fem.fem import FEMCoefficient
from . import util, time
from ..models.common import DirichletBoundary


modelList = {}

def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mat):
    return CompositeModel([modelList[m.className](m, mat) for m in fem.models], mat)


class NGSModel:
    def __init__(self, model, vars, addVariables=False):
        self._model = model
        self._funcs = vars
        self._vars = []

        if addVariables:
            for eq in model.equations:
                self.addVariable(eq.variableName, eq.variableDimension, region=eq.geometries, order=model.order, isScalar=eq.isScalar, fetype=model.type)

    def addVariable(self, name, vdim, dirichlet="auto", initialValue="auto", initialVelocity=None, region=None, order=1, isScalar=False, type="x", fetype="H1"):
        initialValue = self._funcs[self.__initialValue(vdim, initialValue)]
        if initialValue is None:
            raise RuntimeError("Invalid initial value for " + str(name))
            
        if initialVelocity is None:
            initialVelocity = self._funcs[FEMCoefficient([0]*vdim)]

        kwargs = {"fetype": fetype, "isScalar": isScalar, "order": order, "size": vdim}
        if region is not None:
            if region.selectionType() == "Selected":
                kwargs["definedon"] = "|".join([region.geometryType.lower() + str(r) for r in region])

        if dirichlet == "auto":
            dirichlet = self._model.boundaryConditions.coef(DirichletBoundary)
        dirichlet = self.__dirichlet(dirichlet, vdim)

        if dirichlet is not None:
            kwargs["dirichlet"] = ["|".join(["boundary" + str(item) for item in dirichlet[i]]) for i in range(vdim)]

        fes = util.FunctionSpace(**kwargs)
        self._vars.append(util.NGSVariable(name, fes, initialValue, initialVelocity, type=type))

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
    
    def discretize(self, tnt, sols, dti):
        d = {}
        for v in self.variables:
            trial = tnt[v][0]
            if self._model.discretization == "ForwardEuler":
                d.update(time.BackwardEuler.generateWeakforms(v, trial, sols, dti))
            elif self._model.discretization == "BackwardEuler":
                d.update(time.BackwardEuler.generateWeakforms(v, trial, sols, dti))
            elif self._model.discretization == "BDF2":
                d.update(time.BDF2.generateWeakforms(v, trial, sols, dti))
            elif self._model.discretization == "NewmarkBeta":
                d.update(time.NewmarkBeta.generateWeakforms(v, trial, sols, dti))
            else:
                raise RuntimeError("Unknown discretization: "+self._model.discretization)
        return d

    def updater(self, tnt, sols, dti):
        d = self.discretize(tnt, sols, dti)
        res = {}
        for v in self.variables:
            trial = tnt[v][0]
            if trial.t in d:
                res[trial.t] = d[trial.t]
            if trial.tt in d:
                res[trial.tt] = d[trial.tt]
        return res

    @property
    def variables(self):
        return self._vars

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
        tnt = {v.name: tnt for v, tnt in self.TnT.items()}
        wf = sum([model.weakform(tnt, self._mat) for model in self._models])
        if type == "discretized":
            wf = self.__discretizeWeakform(wf, sols, symbols)
        if type == "initial":
            wf = self.__initial(wf, sols)
        return wf

    def __discretizeWeakform(self, wf, sols, symbols):
        d = dict(self.discretize(sols))
        if symbols is not None:
            for v, (trial, test) in self.TnT.items():
                if v.name not in symbols:
                    d.update({trial: sols.X(v), trial.t: sols.V(v), trial.tt: sols.A(v), test:0, util.grad(test): 0})
        return wf.replace(d)

    def __initial(self, wf, sols):
        d = {}
        for v, (trial, test) in self.TnT.items():
            d.update({trial: sols.X(v), trial.t: sols.V(v), trial.tt: trial, util.grad(trial): util.grad(sols.X(v))})
        return wf.replace(d)
    
    def discretize(self, sols):
        d = {}
        for m in self._models:
            d.update(m.discretize(self.TnT, sols, self._mat.const.dti))
        return d

    def updater(self, sols):
        d = {}
        for m in self._models:
            d.update(m.updater(self.TnT, sols, self._mat.const.dti))
        return d

    def initialValue(self, fes):
        x = fes.gridFunction([c for v in self.variables for c in v.value(fes)])
        v = fes.gridFunction([c for v in self.variables for c in v.velocity(fes)])
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
    
    @property
    def TnT(self):
        return {v: (util.TrialFunction(v), util.TestFunction(v)) for v in self.variables}
    
