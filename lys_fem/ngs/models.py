import ngsolve
from lys_fem.fem import FEMCoefficient
from . import util
from ..models.common import DirichletBoundary


modelList = {}

def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return CompositeModel(mesh, [modelList[m.className](m, mesh, mat) for m in fem.models], mat)


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
        coef = self._init/self._scale
        if self.size == 1:
            return [coef]
        else:
            return [coef[i] for i in range(coef.shape[0])]
    
    @property
    def velocity(self):
        coef = self._vel/self._scale
        if self.size == 1:
            return [coef]
        else:
            return [coef[i] for i in range(coef.shape[0])]

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
    def __init__(self, model, mesh, vars, addVariables=False, order=1):
        self._model = model
        self._mesh = mesh
        self._funcs = vars
        self._vars = []

        if addVariables:
            for eq in model.equations:
                self.addVariable(eq.variableName, eq.variableDimension, region=eq.geometries, order=order, isScalar=eq.isScalar)

    def addVariable(self, name, vdim, dirichlet="auto", initialValue="auto", initialVelocity=None, region=None, order=1, isScalar=False):
        initialValue = self._funcs.eval(self.__initialValue(vdim, initialValue))
        if initialValue is None:
            raise RuntimeError("Invalid initial value for " + str(name))
            
        if initialVelocity is None:
            initialVelocity = self._funcs.eval(FEMCoefficient([0]*vdim))

        kwargs = {"order": order}
        if region is not None:
            if region.selectionType() == "Selected":
                kwargs["definedon"] = "|".join([region.geometryType.lower() + str(r) for r in region])

        if dirichlet == "auto":
            dirichlet = self._model.boundaryConditions.coef(DirichletBoundary)
        dirichlet = self.__dirichlet(dirichlet, vdim)

        fess = []
        for i in range(vdim):
            if dirichlet is not None:
                kwargs["dirichlet"] = "|".join(["boundary" + str(item) for item in dirichlet[i]])
            fess.append(ngsolve.H1(self._mesh, **kwargs))

        self._vars.append(NGSVariable(name, fess, self.scale, initialValue, initialVelocity, xscale=self._mesh.scale, isScalar=isScalar))

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

    def coef(self, cls, name="Undefined"):
        c = self._model.boundaryConditions.coef(cls)
        if c is not None:
            return util.NGSFunction(self._funcs.eval(c), name=name)
        c = self._model.domainConditions.coef(cls)
        if c is not None:
            return util.NGSFunction(self._funcs.eval(c), name=name)
        return util.NGSFunction()

    @property
    def variables(self):
        return self._vars

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
        self.__initTnT()

    def __initTnT(self):
        if not isinstance(self._fes, ngsolve.ProductSpace):
            trial, test = [[t] for t in self._fes.TnT()]
        else:
            trial, test = self._fes.TnT()
        n = 0
        for var in self.variables:
            var.setTnT(trial[n:n+var.size], test[n:n+var.size])
            n+=var.size

    def weakforms(self):
        util.dx.setScale(self._mesh.scale)
        util.dx.setMesh(self._mesh)
        util.ds.setMesh(self._mesh)
        tnt = {var.name: (var.trial, var.test) for var in self.variables}   # trial and test functions
        self._mat.update({v.name: v.trial for v in self.variables})         # update variable dictionary
        return sum([model.weakform(tnt, self._mat) for model in self._models], util.NGSFunction())

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
    