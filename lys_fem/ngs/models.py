from ngsolve import BilinearForm, LinearForm, GridFunction, H1, Parameter, VectorH1

from . import util


modelList = {}
dti = Parameter(0.0)

def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return CompositeModel(mesh, [modelList[m.className](m, mesh, mat) for m in fem.models])


class NGSModel:
    def __init__(self, model, mesh, addVariables=False):
        self._model = model
        self._mesh = mesh

        self._fes = []
        self._vnames = []
        self._sol = []

        if addVariables:
            for eq in model.equations:
                self.addVariable(eq.variableName, eq.variableDimension, "auto", "auto", eq.geometries)      

    def addVariable(self, name, vdim, dirichlet=None, initialValue=None, region=None, order=1):
        if initialValue is None:
            initialValue = util.generateCoefficient([0]*vdim)
        elif initialValue == "auto":
            initialValue = util.generateGeometryCoefficient(self._mesh, self._model.initialConditions)

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

        sol = GridFunction(fes)
        sol.Set(initialValue)

        self._fes.append(fes)
        self._vnames.append(name)
        self._sol.append(sol)

    @property
    def spaces(self):
        return self._fes

    @property
    def mesh(self):
        return self._mesh

    @property
    def variableNames(self):
        return self._vnames

    @property
    def sol(self):
        return self._sol

    @property
    def isNonlinear(self):
        return False

    def TnT(self):
        return util.prod(self.spaces).TnT()

    @property
    def name(self):
        return self._model.name


class CompositeModel:
    def __init__(self, mesh, models):
        self._mesh = mesh
        self._models = models
        self._fes = self.space

        self._bilinear = BilinearForm(self._fes)
        self._bilinear += self.bilinearform

        self._linear = LinearForm(self._fes)
        self._linear += self.linearform

        self._dti_prev = None

    def solve(self, solver, dt_inv=0):
        dti.Set(dt_inv)
        if not hasattr(self, "_x"):
            self._x = self.__getSolution()

        if (self._dti_prev != dt_inv) and (not self.isNonlinear):
            self._bilinear.Assemble()
            self._dti_prev = dt_inv

        self._linear.Assemble()
        solver.solve(self, self._x)
        
        self.__setSolution(self._x)
        return self.solution
    
    def __getSolution(self):
        sols = self._sols
        u = GridFunction(self._fes)
        if len(sols) == 1:
            u.Set(*sols)
        else:
            for ui, i in zip(u.components, sols):
                ui.Set(i)
        return u
    
    def __setSolution(self, x):
        # Set respective grid functions
        sols = self._sols
        if len(sols) == 1:
            sols[0].Set(x)
        else:
            for si, i in zip(sols, x.components):
                si.Set(i)

    def __call__(self, x):
        self._linear.Assemble()
        if self.isNonlinear:
            return self._bilinear.Apply(x.vec) - self._linear.vec 
        else:
            return self._bilinear.mat * x.vec - self._linear.vec

    def Jacobian(self, x):
        if self.isNonlinear:
            self._bilinear.AssembleLinearization(x.vec)
        return self._bilinear.mat.Inverse(self._fes.FreeDofs(), "pardiso")

    @property
    def bilinearform(self):
        return sum(m.bilinearform for m in self._models)

    @property
    def linearform(self):
        return sum(m.linearform for m in self._models)

    @property
    def space(self):
        spaces = []
        for m in self._models:
            spaces.extend(m.spaces)
        return util.prod(spaces)
    
    @property
    def isNonlinear(self):
        for m in self._models:
            if m.isNonlinear:
                return True
        return False

    @property
    def _sols(self):
        sols = []
        for m in self._models:
            sols.extend(m.sol)
        return sols

    @property
    def solution(self):
        result = {}
        for m in self._models:
            for name, sol in zip(m.variableNames, m.sol):
                if len(sol.shape) == 0:
                    result[name] = sol.vec
                else:
                    for i, s in enumerate(sol.components):
                        result[name+str(i+1)] = s.vec
        return result

    @property
    def models(self):
        return self._models