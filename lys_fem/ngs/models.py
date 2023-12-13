import numpy as np
from ngsolve import BilinearForm, LinearForm, GridFunction, H1

from ..fem import DirichletBoundary
from . import util


modelList = {}


def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return [modelList[m.name](m, mesh, mat) for m in fem.models]


class NGSModel:
    def __init__(self, model, mesh, order=1):
        self._model = model
        self._mesh = mesh

        self._fes = [H1(mesh, order=order, dirichlet=d) for d in self._dirichletCondition]
        if self._model.variableDimension() == 1:
            self._vnames = [self._model.variableName]
        else:
            self._vnames = [self._model.variableName + str(i+1) for i in range(self._model.variableDimension)]
        self._init = util.generateDomainCoefficient(mesh, self._model.initialConditions)

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
    def initialValues(self):
        return [self._init]

    def TnT(self):
        if self._model.variableDimension() == 1:
            return self._fes[0].TnT()

    @property
    def _dirichletCondition(self):
        conditions = self._model.boundaryConditions.get(DirichletBoundary)
        bdr_dir = {i: [] for i in range(self._model.variableDimension())}
        for b in conditions:
            for axis, check in enumerate(b.components):
                if check:
                    bdr_dir[axis].extend(b.boundaries.getSelection())
        return bdr_dir.values()



class CompositeModel:
    def __init__(self, mesh, models, type):
        self._mesh = mesh
        self._models = models
        self._type = type

        self._fes = self.space
        self._x = self.__getInitialValue()

    def __checkNonlinear(self, wf, trials):
        vars = []
        for trial in trials:
            vars.append(trial)
            for gt in grad(trial):
                vars.append(gt)
        p = sp.poly(wf, *vars)
        for order in p.as_dict().keys():
            if sum(order) > 1:
                return True
        return False

    def __getInitialValue(self):
        # gather all initial values
        inits = []
        for m in self._models:
            inits.extend(m.initialValues)

        # set it to grid function
        u = GridFunction(self._fes)
        if len(inits) == 1:
            u.Set(*inits)
        else:
            for ui, i in zip(u.components, inits):
                ui.Set(i)
        return u

    def solve(self, solver, dt=1):
        fes = self.space
        
        b = BilinearForm(fes)
        b += self.bilinearform

        l = LinearForm(fes)
        l += self.linearform
        b.Assemble()
        l.Assemble()

        res = l.vec.CreateVector()
        res.data = l.vec - b.mat * self._x.vec
        self._x.vec.data += b.mat.Inverse(fes.FreeDofs()) * res

        return self.solution

    def __call__(self, x):
        K, b = self.K, self.b
        res = mfem.Vector(x.Size())
        K.Mult(x, res)
        res -= b
        return res

    def grad(self, x):
        return self._J

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
        result = spaces[0]
        for sp in spaces[1:]:
            result = result * sp
        return result

    @property
    def isNonlinear(self):
        return self._nonlinear

    @property
    def solution(self):
        names = []
        for m in self._models:
            names.extend(m.variableNames)
        if len(names) == 1:
            return {names[0]: np.array(self._x.vec)}
        else:
            return {n: sol.vec for n, sol in zip(names, self._x.components)}

 