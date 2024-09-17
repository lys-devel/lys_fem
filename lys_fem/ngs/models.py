import ngsolve
from . import util

modelList = {}

def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return CompositeModel(mesh, [modelList[m.className](m, mesh) for m in fem.models], mat)


class NGSVariable:
    def __init__(self, name, fes, scale, initialValue, initialVelocity, xscale, isScalar):
        self._name = name
        self._fes = fes
        self._scale = scale
        self._init = initialValue
        self._vel = initialVelocity
        self._isScalar = isScalar
        self._xscale = xscale

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
            return [self._init/self._scale]
        else:
            return [self._init[i]/self._scale for i in range(self._init.shape[0])]
    
    @property
    def velocity(self):
        if self.size == 1:
            return [self._vel]
        else:
            return [self._vel[i] for i in range(self._vel.shape[0])]

    def setTnT(self, trial, test):
        if self.size==1 and self._isScalar:
            trial, test = trial[0], test[0]        
        self._trial = util.TrialFunction(self.name, trial, xscale=self._xscale, scale=self._scale)
        self._test = util.TestFunction(test, name=self.name, xscale=self._xscale)
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
                self.addVariable(eq.variableName, eq.variableDimension, region=eq.geometries, order=order, isScalar=eq.isScalar)

    def addVariable(self, name, vdim, dirichlet="auto", initialValue="auto", initialVelocity=None, region=None, order=1, isScalar=False):
        initialValue = self.__initialValue(vdim, initialValue)
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
            fess.append(ngsolve.H1(self._mesh, **kwargs))

        self._vars.append(NGSVariable(name, fess, self.scale, initialValue, initialVelocity, xscale=self._mesh.scale, isScalar=isScalar))

    def __initialValue(self, vdim, initialValue):
        if initialValue is None:
            return util.generateCoefficient([0]*vdim)
        if initialValue != "auto":
            return initialValue
        init = None
        for type in self._model.initialConditionTypes:
            c = self._model.initialConditions.coef(type)
            if init is None:
                init = c
            else:
                init.update(c)
        initialValue = util.generateCoefficient(init, self._mesh)
        return initialValue

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
    def name(self):
        return self._model.name
    
    @property
    def scale(self):
        return 1


class CompositeModel:
    def __init__(self, mesh, models, mat):
        self._mesh = mesh
        self._models = models
        self._mat = mat
        self._fes = util.prod([v.finiteElementSpace for v in self.variables])

    def weakforms(self):
        # prepare test and trial functions
        tnt = {}
        if not isinstance(self.finiteElementSpace, ngsolve.ProductSpace):
            trial, test = [[t] for t in self._fes.TnT()]
        else:
            trial, test = self._fes.TnT()
        n = 0
        for var in self.variables:
            tnt[var.name] = var.setTnT(trial[n:n+var.size], test[n:n+var.size])
            n+=var.size

        util.dx.setScale(self._mesh.scale)

        # create weakforms
        wf = util.NGSFunction()
        for model in self._models:
            wf += model.weakform(tnt, self._mat)
        return wf

    def initialValue(self, use_a=True):
        x = util.GridFunction(self._fes, [c for v in self.variables for c in v.value])
        v = util.GridFunction(self._fes, [c for v in self.variables for c in v.velocity])
        a = None
        if use_a:
            fes = self.finiteElementSpace
            wf = self.weakforms()

            d = {}
            for var in self.variables:
                d[var.trial.t] = util.NGSFunction()
                d[var.trial.tt] = util.NGSFunction()
            wf_K = wf.replace(d)
            K = ngsolve.BilinearForm(fes)
            K += wf_K.lhs.eval()

            d = {}
            for var in self.variables:
                d[util.grad(var.trial)] = util.NGSFunction()
                d[var.trial] = util.NGSFunction()
                d[var.trial.t] = var.trial
                d[var.trial.tt] = util.NGSFunction()
            wf_C = wf.replace(d)
            C = ngsolve.BilinearForm(fes)
            C += wf_C.lhs.eval()

            d = {}
            for var in self.variables:
                d[util.grad(var.trial)] = util.NGSFunction()
                d[var.trial] = util.NGSFunction()
                d[var.trial.t] = util.NGSFunction()
                d[var.trial.tt] = var.trial
            wf_M = wf.replace(d)
            M = ngsolve.BilinearForm(fes)
            M += wf_M.lhs.eval()

            d = {}
            for var in self.variables:
                d[var.trial.t] = var.trial
                d[var.trial.tt] = var.trial
            wf_F = wf.replace(d)
            F = ngsolve.LinearForm(fes)
            F += wf_F.rhs.eval()

            rhs = - F.vec - K.Apply(x.vec) - C.Apply(v.vec)
            M.AssembleLinearization(x.vec)

            a = util.GridFunction(fes)
            a.vec.data  = M.mat.Inverse(fes.FreeDofs(), "pardiso") * rhs
        return x, v, a
    
    @property
    def models(self):
        return self._models
    
    @property
    def finiteElementSpace(self):
        return self._fes

    @property
    def variables(self):
        return sum([m.variables for m in self._models], [])
    