import ngsolve
from .util import prod


class FunctionSpace:
    def __init__(self, fetype, size=1, dirichlet=None, isScalar=True, **kwargs):
        self._type = fetype
        self._size = size
        self._scalar = isScalar
        if dirichlet is None:
            dirichlet = [None] * size
        self._dirichlet = dirichlet
        self._kwargs = dict(kwargs)

    @property
    def size(self):
        return self._size
    
    @property
    def isScalar(self):
        return self._scalar and self._size==1

    def eval(self, mesh):
        fess = []
        if self._type == "H1":
            space = ngsolve.H1
        elif self._type == "L2":
            space = ngsolve.L2
        for i in range(self.size):
            if self._dirichlet[i] is None:
                fess.append(space(mesh, **self._kwargs))
            else:
                fess.append(space(mesh, dirichlet=self._dirichlet[i], **self._kwargs))
        return prod(fess)


class H1(FunctionSpace):
    def __init__(self, **kwargs):
        super().__init__("H1", **kwargs)


class L2(FunctionSpace):
    def __init__(self, **kwargs):
        super().__init__("L2", **kwargs)


class NGSVariable:
    def __init__(self, name, fes, initialValue=None, initialVelocity=None, type="x"):
        self._name = name
        self._fes = fes
        self._init = initialValue
        self._vel = initialVelocity
        self._type = type

    @property
    def name(self):
        return self._name
    
    @property
    def size(self):
        return self._fes.size
    
    @property
    def type(self):
        return self._type

    @property
    def isScalar(self):
        return self._fes.isScalar

    def finiteElementSpace(self, mesh):
        return self._fes.eval(mesh)
    
    def value(self, fes):
        coef = self._init
        if coef.valid:
            coef = coef.eval(fes)
        else:
            coef = ngsolve.CoefficientFunction(tuple([0] * self.size))
        if self.size == 1:
            return [coef]
        else:
            return [coef[i] for i in range(coef.shape[0])]
    
    def velocity(self, fes):
        coef = self._vel
        if coef.valid:
            coef = coef.eval(fes)
        else:
            coef = ngsolve.CoefficientFunction(tuple([0] * self.size))
        if self.size == 1:
            return [coef]
        else:
            return [coef[i] for i in range(coef.shape[0])]
