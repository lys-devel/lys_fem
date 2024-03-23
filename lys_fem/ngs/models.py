import numpy as np
from ngsolve import BilinearForm, LinearForm, GridFunction, H1, Parameter, VectorH1

from . import util


modelList = {}
dti = Parameter(0.0)

def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return CompositeModel(mesh, [modelList[m.className](m, mesh, mat) for m in fem.models], mat)


class NGSVariable:
    def __init__(self, name, fes, scale, initialValue):
        self._name = name
        self._fes = fes
        self._scale = scale
        self._sol = GridFunction(fes)
        self._sol.Set(initialValue)

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
    def sol(self):
        return self._sol


class NGSModel:
    def __init__(self, model, mesh, addVariables=False, order=1):
        self._model = model
        self._mesh = mesh
        self._vars = []

        if addVariables:
            for eq in model.equations:
                self.addVariable(eq.variableName, eq.variableDimension, "auto", "auto", eq.geometries, order=order)

    def addVariable(self, name, vdim, dirichlet=None, initialValue=None, region=None, order=1, scale=1):
        if initialValue is None:
            initialValue = util.generateCoefficient([0]*vdim)
            scale = 1
        elif initialValue == "auto":
            init = self._model.initialConditions.coef(self._model.initialConditionTypes[0])
            scale = init.scale
            initialValue = util.generateCoefficient(init, self._mesh)

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

        self._vars.append(NGSVariable(name, fes, scale, initialValue))

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

        vnames = [v.name for v in self.variables]
        if len(vnames) == 1:
            tnt = {vnames[0]: self._fes.TnT()}
        else:
            tnt = {name: (test, trial) for name, test, trial in zip(vnames, *self._fes.TnT())}
        sols = {name: sol for name, sol in zip(vnames, self._sols)}

        self._bilinear = BilinearForm(self._fes)
        self._bilinear += sum([m.bilinearform(tnt, sols) for m in self._models])

        self._linear = LinearForm(self._fes)
        self._linear += sum([m.linearform(tnt, sols) for m in self._models])

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
        if len(self.variables) == 1:
            self.variables[0].sol.Set(x)
        else:
            for i, v in enumerate(self.variables):
                v.sol.Set(x.components[i])

    def __call__(self, x):
        self._linear.Assemble()
        if self.isNonlinear:
            self._bilinear.Assemble()
            F = self._bilinear.Apply(x.vec) - self._linear.vec
            return self._bilinear.Apply(x.vec) - self._linear.vec 
        else:
            return self._bilinear.mat * x.vec - self._linear.vec

    def Jacobian(self, x):
        if self.isNonlinear:
            self._bilinear.AssembleLinearization(x.vec)
        return self._bilinear.mat.Inverse(self._fes.FreeDofs(), "pardiso")
    
    @property
    def isNonlinear(self):
        for m in self._models:
            if m.isNonlinear:
                return True
        return False

    @property
    def _sols(self):
        return [v.sol for v in self.variables]

    @property
    def solution(self):
        result = {}
        for v in self.variables:
            if len(v.sol.shape) == 0:
                result[v.name] = np.array(v.sol.vec) * v.scale
            else:
                result[v.name] = [np.array(s.vec) * v.scale for s in v.sol.components]

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

    @property
    def models(self):
        return self._models
    
    @property
    def variables(self):
        return sum([m.variables for m in self._models], [])