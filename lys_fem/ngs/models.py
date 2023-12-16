from ngsolve import BilinearForm, LinearForm, GridFunction, H1, Parameter, VectorH1

from ..fem import DirichletBoundary
from . import util


modelList = {}
dti = Parameter(0.0)

def addNGSModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return [modelList[m.name](m, mesh, mat) for m in fem.models]


class NGSModel:
    def __init__(self, model, mesh, order=1):
        self._model = model
        self._mesh = mesh

        vdim = self._model.variableDimension()
        dirichlet = self._dirichletCondition
        if vdim == 1:
            self._fes = [H1(mesh, order=order, dirichlet=dirichlet[0])]
        elif vdim==2:
            self._fes = [VectorH1(mesh, order=order, dirichletx=dirichlet[0], dirichlety=dirichlet[1])]
        elif vdim==3:
            self._fes = [VectorH1(mesh, order=order, dirichletx=dirichlet[0], dirichlety=dirichlet[1], dirichletz=dirichlet[2])]
        self._vnames = [self._model.variableName]

        sol = GridFunction(self._fes[0])
        sol.Set(util.generateDomainCoefficient(mesh, self._model.initialConditions))
        self._sol = [sol]

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

    def TnT(self):
        return util.prod(self.spaces).TnT()

    @property
    def _dirichletCondition(self):
        conditions = self._model.boundaryConditions.get(DirichletBoundary)
        bdr_dir = {i: [] for i in range(self._model.variableDimension())}
        for b in conditions:
            for axis, check in enumerate(b.components):
                if check:
                    bdr_dir[axis].extend(b.boundaries.getSelection())
        return ["|".join([str(item) for item in value]) for value in bdr_dir.values()]



class CompositeModel:
    def __init__(self, mesh, models):
        self._mesh = mesh
        self._models = models
        self._fes = self.space

        self._bilinear = BilinearForm(self._fes)
        self._bilinear += self.bilinearform

        self._linear = LinearForm(self._fes)
        self._linear += self.linearform

    def solve(self, solver, dt_inv=0):
        fes = self.space
        dti.Set(dt_inv)

        if not hasattr(self, "_x"):
            self._x = self.__getSolution()
        
        self._bilinear.Assemble()
        self._linear.Assemble()

        res = self._linear.vec.CreateVector()
        res.data = self._linear.vec - self._bilinear.mat * self._x.vec
        self._x.vec.data += self._bilinear.mat.Inverse(fes.FreeDofs(), "pardiso") * res

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

 