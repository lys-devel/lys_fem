from ngsolve import GridFunction, H1, VectorH1, dx

from . import util

modelList = {}

def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return CompositeModel(mesh, [modelList[m.className](m, mesh, mat) for m in fem.models])


class NGSVariable:
    def __init__(self, name, fes, scale, initialValue, initialVelocity):
        self._name = name
        self._fes = fes
        self._scale = scale
        self._init = initialValue
        self._vel = initialVelocity

    @property
    def name(self):
        return self._name

    @property
    def finiteElementSpace(self):
        return self._fes
    
    @property
    def scale(self):
        return self._scale
    
    @property
    def value(self):
        return self._init
    
    @property
    def velocity(self):
        return self._vel

    def setTnT(self, trial, test):
        self._trial = util.TrialFunction(self.name, trial)
        self._test = util.TestFunction(test, name=self.name)

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
                self.addVariable(eq.variableName, eq.variableDimension, "auto", "auto", None, eq.geometries, order=order)

    def addVariable(self, name, vdim, dirichlet=None, initialValue=None, initialVelocity=None, region=None, order=1, scale=1):
        if initialValue is None:
            initialValue = util.generateCoefficient([0]*vdim)
            scale = 1
        elif initialValue == "auto":
            init = self._model.initialConditions.coef(self._model.initialConditionTypes[0])
            scale = init.scale
            initialValue = util.generateCoefficient(init, self._mesh)
        if initialVelocity is None:
            initialVelocity = util.generateCoefficient([0]*vdim)

        kwargs = {}
        if dirichlet == "auto":
            dirichlet = util.generateDirichletCondition(self._model)
        if dirichlet is not None:
            if vdim == 1:
                kwargs["dirichlet"] = "|".join(["boundary" + str(item) for item in dirichlet[0]])
            if vdim  > 1:
                kwargs["dirichletx"] = "|".join(["boundary" + str(item) for item in dirichlet[0]])
                kwargs["dirichlety"] = "|".join(["boundary" + str(item) for item in dirichlet[1]])
            if vdim == 3:
                kwargs["dirichletz"] = "|".join(["boundary" + str(item) for item in dirichlet[2]])

        if region is not None:
            if region.selectionType() == "Selected":
                kwargs["definedon"] = "|".join([region.geometryType.lower() + str(r) for r in region])

        if vdim == 1:
            fes = H1(self._mesh, order=order, **kwargs)
        elif vdim==2:
            fes = VectorH1(self._mesh, order=order, **kwargs)
        elif vdim==3:
            fes = VectorH1(self._mesh, order=order, **kwargs)

        self._vars.append(NGSVariable(name, fes, scale, initialValue, initialVelocity))

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
    def __init__(self, mesh, models):
        self._mesh = mesh
        self._models = models
        self._fes = util.prod([v.finiteElementSpace for v in self.variables])

    def weakforms(self):
        # prepare test and trial functions
        if len(self.variables) == 1:
            self.variables[0].setTnT(*self._fes.TnT())
        else:
            for var, trial, test in zip(self.variables, *self._fes.TnT()):
                var.setTnT(trial, test)

        # create weakforms
        wf = util.NGSFunction()
        for model in self._models:
            wf += model.weakform({v.name: v for v in self.variables})
        return wf
    
    @property
    def initialValue(self):
        return self.__getGridFunction([v.value for v in self.variables])

    @property
    def initialVelocity(self):
        return self.__getGridFunction([v.velocity for v in self.variables])

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
    
