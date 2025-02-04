import ngsolve
from .util import prod, GridFunction


class FunctionSpace:
    def __init__(self, fetype="H1", size=1, dirichlet=None, isScalar=True, **kwargs):
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
        if self._type == "H1":
            space = ngsolve.H1
        elif self._type == "L2":
            space = ngsolve.L2

        fess = []
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


class FiniteElementSpace:
    def __init__(self, vars, mesh, symbols=None, symmetric=False, condense=False, jacobi=None):
        self._mesh = mesh
        self._vars = vars
        self._symbols = symbols
        self._symmetric = symmetric
        self._condense = condense
        self._jacobi = jacobi
        if symbols is None:
            self._fes = prod([v.finiteElementSpace(mesh) for v in vars])
            self._tnt = self.__TnT_dict(vars, self._fes)
        else:
            self._fes_glb = prod([v.finiteElementSpace(mesh) for v in vars])
            self._fes = prod([v.finiteElementSpace(mesh) for v in vars if v.name in symbols])
            self._tnt = self.__TnT_dict([v for v in vars if v.name in symbols], self._fes)
            self._mask = self.__mask(self._fes_glb, symbols)

    def __TnT_dict(self, vars, fes):
        if isinstance(fes, (ngsolve.ProductSpace, ngsolve.comp.Compress)):
            trials, tests = fes.TnT()
        else:
            trials, tests = [[t] for t in fes.TnT()]

        n = 0
        res = {}
        for var in vars:
            if var.isScalar:
                trial, test = trials[n], tests[n]
            else:
                trial, test = trials[n:n+var.size], tests[n:n+var.size]
            res[var] = (trial, test)
            n+=var.size
        return res

    def __mask(self, fes, symbols):
        dofs = ngsolve.BitArray(fes.FreeDofs())
        n = 0
        for v in self.variables:
            for j in range(n, n+v.size):
                dofs[fes.Range(j)]=v.name in symbols
            n += v.size
        return dofs

    @property
    def mesh(self):
        return self._mesh
    
    @property
    def variables(self):
        return self._vars
    
    @property
    def mask(self):
        return self._mask
    
    @property
    def J(self):
        return self._jacobi
    
    def gridFunction(self, value=None):
        return GridFunction(self, value)
    
    @property
    def ndof(self):
        return self._fes.ndofglobal
    
    def trial(self, var):
        return self._tnt[var][0]

    def test(self, var):
        return self._tnt[var][1]
    
    def BilinearForm(self):
        return ngsolve.BilinearForm(self._fes, condense=self._condense, symmetric=self._symmetric)

    def LinearForm(self):
        return ngsolve.LinearForm(self._fes)
    
    def FreeDofs(self):
        return self._fes.FreeDofs(self._condense)