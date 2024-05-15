from ngsolve import GridFunction, H1, ProductSpace, CoefficientFunction

from . import util

modelList = {}

def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return CompositeModel(mesh, [modelList[m.className](m, mesh) for m in fem.models], mat)


class NGSVariable:
    def __init__(self, name, fes, scale, initialValue, initialVelocity, isScalar):
        self._name = name
        self._fes = fes
        self._scale = scale
        self._init = initialValue
        self._vel = initialVelocity
        self._isScalar = isScalar

    @property
    def name(self):
        return self._name
    
    @property
    def size(self):
        return len(self._fes)

    @property
    def isScalar(self):
        return self._isScalar

    @property
    def finiteElementSpace(self):
        return util.prod(self._fes)
    
    @property
    def scale(self):
        return self._scale
    
    @property
    def value(self):
        if self.size == 1:
            return [self._init]
        else:
            return [self._init[i] for i in range(self._init.shape[0])]
    
    @property
    def velocity(self):
        if self.size == 1:
            return [self._vel]
        else:
            return [self._vel[i] for i in range(self._vel.shape[0])]

    def setTnT(self, trial, test):
        if self.size==1:
            if self._isScalar and not isinstance(trial, CoefficientFunction):
                trial = trial[0]
                test = test[0]
            if not self._isScalar and isinstance(trial, CoefficientFunction):
                trial = [trial]
                test = [test]
        
        self._trial = util.TrialFunction(self.name, trial)
        self._test = util.TestFunction(test, name=self.name)
        return self._trial, self._test

    @property
    def trial(self):
        return self._trial

    @property
    def test(self):
        return self._test


class NGSModel:
    def __init__(self, model, mesh, addVariables=False, order=1):
        self._model = model
        self._mesh = mesh
        self._vars = []

        if addVariables:
            for eq in model.equations:
                self.addVariable(eq.variableName, eq.variableDimension, "auto", "auto", None, region=eq.geometries, order=order, isScalar=eq.isScalar)

    def addVariable(self, name, vdim, dirichlet=None, initialValue=None, initialVelocity=None, region=None, order=1, scale=1, isScalar=False):
        if initialValue is None:
            initialValue = util.generateCoefficient([0]*vdim)
            scale = 1
        elif initialValue == "auto":
            init = self._model.initialConditions.coef(self._model.initialConditionTypes[0])
            scale = init.scale
            initialValue = util.generateCoefficient(init, self._mesh)
        if initialVelocity is None:
            initialVelocity = util.generateCoefficient([0]*vdim)

        kwargs = {"order": order}
        if region is not None:
            if region.selectionType() == "Selected":
                kwargs["definedon"] = "|".join([region.geometryType.lower() + str(r) for r in region])

        if dirichlet == "auto":
            dirichlet = util.generateDirichletCondition(self._model)

        fess = []
        for i in range(vdim):
            if dirichlet is not None:
                kwargs["dirichlet"] = "|".join(["boundary" + str(item) for item in dirichlet[i]])
            fess.append(H1(self._mesh, **kwargs))

        self._vars.append(NGSVariable(name, fess, scale, initialValue, initialVelocity, isScalar=isScalar))

    def coef(self, cls, name="Undefined"):
        c = self._model.boundaryConditions.coef(cls)
        if c is not None:
            return util.coef(c, self.mesh, name=name)
        c = self._model.domainConditions.coef(cls)
        if c is not None:
            return util.coef(c, self.mesh, name=name)
        return util.NGSFunction()

    @property
    def variables(self):
        return self._vars

    @property
    def mesh(self):
        return self._mesh

    @property
    def isNonlinear(self):
        return False

    @property
    def name(self):
        return self._model.name


class CompositeModel:
    def __init__(self, mesh, models, mat):
        self._mesh = mesh
        self._models = models
        self._mat = mat
        self._fes = util.prod([v.finiteElementSpace for v in self.variables])

    def weakforms(self):
        # prepare test and trial functions
        tnt = {}
        if self.isSingle:
            trial, test = [[t] for t in self._fes.TnT()]
        else:
            trial, test = self._fes.TnT()
        n = 0
        for var in self.variables:
            tnt[var.name] = var.setTnT(trial[n:n+var.size], test[n:n+var.size])
            n+=var.size

        # create weakforms
        wf = util.NGSFunction()
        for model in self._models:
            wf += model.weakform(tnt, self._mat)
        return wf
    
    @property
    def initialValue(self):
        return self.__getGridFunction([c for v in self.variables for c in v.value])

    @property
    def initialVelocity(self):
        return self.__getGridFunction([c for v in self.variables for c in v.velocity])

    def __getGridFunction(self, sols):
        u = GridFunction(self._fes)
        if len(sols) == 1:
            u.Set(*sols)
        else:
            for ui, i in zip(u.components, sols):
                ui.Set(i)
        return u
        
    @property
    def isNonlinear(self):
        for m in self._models:
            if m.isNonlinear:
                return True
        return False
    
    @property
    def models(self):
        return self._models
    
    @property
    def finiteElementSpace(self):
        return self._fes

    @property
    def variables(self):
        return sum([m.variables for m in self._models], [])
    
    @property
    def isSingle(self):
        return not isinstance(self.finiteElementSpace, ProductSpace)