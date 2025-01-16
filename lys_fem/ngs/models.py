import ngsolve
from lys_fem.fem import FEMCoefficient
from . import util, time
from ..models.common import DirichletBoundary


modelList = {}

def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mat):
    return CompositeModel([modelList[m.className](m, mat) for m in fem.models], mat)


class NGSVariable:
    def __init__(self, name, fes, scale, residualScale, initialValue, initialVelocity, isScalar, type):
        self._name = name
        self._fes = fes
        self._scale = scale
        self._residualScale = residualScale
        self._init = initialValue
        self._vel = initialVelocity
        self._isScalar = isScalar
        self._type = type

    @property
    def name(self):
        return self._name
    
    @property
    def size(self):
        return len(self._fes)
    
    @property
    def type(self):
        return self._type

    @property
    def isScalar(self):
        return self._isScalar

    def finiteElementSpace(self, mesh):
        return util.prod([fes.eval(mesh) for fes in self._fes])
    
    @property
    def scale(self):
        return self._scale

    @property
    def residualScale(self):
        return self._residualScale

    def value(self, fes):
        coef = self._init/self._scale
        if coef.valid:
            coef = coef.eval(fes)
        else:
            coef = ngsolve.CoefficientFunction(tuple([0] * self.size))
        if self.size == 1:
            return [coef]
        else:
            return [coef[i] for i in range(coef.shape[0])]
    
    def velocity(self, fes):
        coef = self._vel/self._scale
        if coef.valid:
            coef = coef.eval(fes)
        else:
            coef = ngsolve.CoefficientFunction(tuple([0] * self.size))
        if self.size == 1:
            return [coef]
        else:
            return [coef[i] for i in range(coef.shape[0])]


class _FESpace:
    def __init__(self, type, kwargs):
        self._type = type
        self._kwargs = dict(kwargs)

    def eval(self, mesh):
        if self._type == "H1":
            return ngsolve.H1(mesh, **self._kwargs)
        if self._type == "L2":
            return ngsolve.L2(mesh, **self._kwargs)


class NGSModel:
    def __init__(self, model, vars, addVariables=False):
        self._model = model
        self._funcs = vars
        self._vars = []

        if addVariables:
            for eq in model.equations:
                self.addVariable(eq.variableName, eq.variableDimension, region=eq.geometries, order=model.order, isScalar=eq.isScalar, fetype=model.type)

    def addVariable(self, name, vdim, dirichlet="auto", initialValue="auto", initialVelocity=None, region=None, order=1, isScalar=False, type="x", scale=None, residualScale=None, fetype="H1"):
        initialValue = self._funcs[self.__initialValue(vdim, initialValue)]
        if initialValue is None:
            raise RuntimeError("Invalid initial value for " + str(name))
            
        if initialVelocity is None:
            initialVelocity = self._funcs[FEMCoefficient([0]*vdim)]

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
            fess.append(_FESpace(fetype, kwargs))
        
        if scale is None:
            scale = self.scale

        if residualScale is None:
            residualScale = self.residualScale

        self._vars.append(NGSVariable(name, fess, scale, residualScale, initialValue, initialVelocity, isScalar=isScalar, type=type))

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

    def weakforms(self):
        tnt = {v.name: tnt for v, tnt in self.TnT.items()}
        return sum([model.weakform(tnt, self._mat) for model in self._models])
    
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

    def initialValue(self, fes, use_a=True):
        x = fes.gridFunction([c for v in self.variables for c in v.value(fes)])
        v = fes.gridFunction([c for v in self.variables for c in v.velocity(fes)])
        a = fes.gridFunction()
        if use_a:
            tnt = self.TnT
            wf = self.weakforms()

            d = {}
            for var, (trial, test) in tnt.items():
                d[trial.t] = 0
                d[trial.tt] = 0
            wf_K = wf.replace(d).lhs
            K = fes.BilinearForm()
            if wf_K.valid:
                K += wf_K.eval(fes)

            d = {}
            for var, (trial, test) in tnt.items():
                d[util.grad(trial)] = 0
                d[trial] = 0
                d[trial.t] = trial
                d[trial.tt] = 0
            wf_C = wf.replace(d).lhs
            C = fes.BilinearForm()
            if wf_C.valid:
                C += wf_C.eval(fes)

            d = {}
            for var, (trial, test) in tnt.items():
                d[util.grad(trial)] = 0
                d[trial] = 0
                d[trial.t] = 0
                d[trial.tt] = trial
            wf_M = wf.replace(d).lhs
            M = fes.BilinearForm()
            if wf_M.valid:
                M += wf_M.eval(fes)

            d = {}
            for var, (trial, test) in tnt.items():
                d[trial.t] = trial
                d[trial.tt] = trial
            wf_F = wf.replace(d).rhs
            F = fes.LinearForm()
            if wf_F.valid:
                F += wf_F.eval(fes)

            rhs = - F.vec - K.Apply(x.vec) - C.Apply(v.vec)
            M.AssembleLinearization(x.vec)

            a.vec.data  = M.mat.Inverse(fes.FreeDofs(), "pardiso") * rhs
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
    
