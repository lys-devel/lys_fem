from ngsolve import GridFunction, H1, VectorH1, dx

from . import util

modelList = {}

def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return CompositeModel(mesh, [modelList[m.className](m, mesh, mat) for m in fem.models], mat)


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
    def __init__(self, mesh, models, mats):
        self._mesh = mesh
        self._models = models
        self._materials = mats
        self._fes = util.prod([v.finiteElementSpace for v in self.variables])

    def weakforms(self, X, V, A):
        vars = {var.name: (xi,vi,ai) for var, xi, vi, ai in zip(self.variables, X, V, A)}

        # prepare test and trial functions
        vnames = [v.name for v in self.variables]
        if len(vnames) == 1:
            tnt = {vnames[0]: self._fes.TnT()}
        else:
            tnt = {name: (trial, test) for name, trial, test in zip(vnames, *self._fes.TnT())}

        # create weakforms
        M, C, K, F = util.generateCoefficient(0)*dx, util.generateCoefficient(0)*dx, util.generateCoefficient(0)*dx, util.generateCoefficient(0)*dx
        for model in self._models:
            m,c,k,f = model.weakform(tnt, vars)
            if m != 0:
                M += m
            if c != 0:
                C += c
            if k!=0:
                K += k
            if f!=0:
                F += f
        return M,C,K,F
    
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
    
    @property
    def materialSolution(self):
        result = {}
        def eval(c):
            sp = H1(self._mesh, order=1)
            gf = GridFunction(sp)
            gf.Set(c)
            return gf.vec
        
        for name, mat in self._materials.items():
            if len(mat.shape) == 0:
                result[name] = eval(mat)
            elif len(mat.shape) == 1:
                result[name] = [eval(mat[i]) for i in range(mat.shape[0])]
            elif len(mat.shape) == 2:
                result[name] = [[eval(mat[i,j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
        return result