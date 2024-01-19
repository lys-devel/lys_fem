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


class NGSModel:
    def __init__(self, model, mesh, addVariables=False, order=1):
        self._model = model
        self._mesh = mesh

        self._fes = []
        self._vnames = []
        self._sol = []
        self._scale = []

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

        sol = GridFunction(fes)
        sol.Set(initialValue)

        self._fes.append(fes)
        self._vnames.append(name)
        self._sol.append(sol)
        self._scale.append(scale)

    def setSolution(self, x):
        if len(self.sol) == 1:
            self.sol[0].Set(x)
        else:
            for si, i in zip(self.sol, x):
                si.Set(i)

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
    def scale(self):
        return self._scale

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
        self._fes = self.space

        if len(self.variableNames) == 1:
            tnt = {self.variableNames[0]: self._fes.TnT()}
        else:
            tnt = {name: (test, trial) for name, test, trial in zip(self.variableNames, *self._fes.TnT())}
        sols = {name: sol for name, sol in zip(self.variableNames, self._sols)}

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
        if len(self.variableNames) == 1:
            self._models[0].setSolution(x)
        else:
            index = 0
            for m in self._models:
                m.setSolution(x.components[index:index+len(m.sol)])
                index += len(m.sol)

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
    def variableNames(self):
        result = []
        for m in self._models:
            result.extend(m.variableNames)
        return result

    @property
    def solution(self):
        result = {}
        for m in self._models:
            for name, sol, scale in zip(m.variableNames, m.sol, m.scale):
                if len(sol.shape) == 0:
                    result[name] = np.array(sol.vec) * scale
                else:
                    result[name] = [np.array(s.vec) * scale for s in sol.components]

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