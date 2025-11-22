import numpy as np
import ngsolve

from .operators import NGSFunctionBase
from .coef import NGSFunction


class GridField(NGSFunctionBase):
    """
    NGSFunction that provide the access to a component of the grid function.
    Args:
        grid(GridFunction): The original grid function
        var(FunctionSpace): The symbol to be used.
    """
    def __init__(self, grid, var):
        self._grid = grid
        self._var = var
        n = 0
        for v in grid.finiteElementSpace.variables:
            if v == var:
                self._n = n
            n += v.size

    def replace(self, d):
        return d.get(self, self)
    
    def eval(self, fes):
        if self._var.isScalar:
            return self._value()
        else:
            return ngsolve.CoefficientFunction(tuple(self._value() + [0] * (3-self._var.size)), dims=self.shape)
        
    def grad(self, fes):
        if self._var.isScalar:
            return self._grad(fes, self._value())
        else:
            v = self._value() + [0] * (3-self._var.size)
            return ngsolve.CoefficientFunction(tuple([self._grad(fes, t) for t in v]), dims=(3, 3)).TensorTranspose((1,0))

    def _grad(self, fes, x):
        if x == 0:
            return ngsolve.CoefficientFunction((0,0,0))
        else:
            g = ngsolve.grad(x)
            g = tuple([g[i] if i < fes.dimension else ngsolve.CoefficientFunction(0) for i in range(3)])
            return ngsolve.CoefficientFunction(g)

    def __contains__(self, item):
        return self == item

    @property
    def shape(self):
        if self._var.isScalar:
            return ()
        else:
            return (3,)
        
    @property
    def isNonlinear(self):
        return False

    @property
    def valid(self):
        return True    
    
    @property
    def hasTrial(self):
        return False
    
    @property
    def rhs(self):
        return self

    @property
    def lhs(self):
        return NGSFunction()
               
    @property
    def isTimeDependent(self):
        return True

    def _value(self):
        if self._var.isScalar:
            return self._grid.components[self._n]
        else:
            return list(self._grid.components[self._n:self._n+self._var.size])

    def __str__(self):
        return self._var.name + "_grid"
    

class RandomFieldFunction(NGSFunctionBase):
    """
    NGSFunction that provide the access to the present solution.
    Args:
        name(str): The symbol name
        type('L2' or 'H1'): The finite element space type.
        tdep(bool): Whether the field is time dependent
    """
    def __init__(self, type, shape, tdep, name=None):
        self._name = name
        self._shape = shape
        self._type = type
        self._tdep = tdep
        self._init = False

    def _initialize(self, fes):
        if self._type=="L2":
            sp = ngsolve.L2(fes.mesh, order=0)
        if self._type=="H1":
            sp = ngsolve.H1(fes.mesh, order=0)
        if self._shape == ():
            self._func = ngsolve.GridFunction(sp)
        else:
            size = 1
            for s in self._shape:
                size *= s
            self._func = [ngsolve.GridFunction(sp) for _ in range(size)]
        self._init = True
        self.update()

    @property
    def shape(self):
        return self._shape

    @property
    def valid(self):
        return True
    
    def update(self):
        if not self._init:
            raise RuntimeWarning("The random field is not initialized. Please delete unused field.")
        if self._shape == ():
            self._func.vec.data = np.random.normal(size=len(self._func.vec))
        else:
            for f in self._func:
                f.vec.data = np.random.normal(size=len(f.vec))

    def eval(self, fes):
        if not self._init:
            self._initialize(fes)
        if self._shape == ():
            return self._func
        else:
            return ngsolve.CoefficientFunction(tuple(self._func), dims=self._shape)
            
    def grad(self, fes):
        if not self._init:
            self._initialize(fes)
        if self._shape == ():
            return ngsolve.grad(self._func)
        else:
            raise RuntimeError("grad for multi-dim random is not defined.")
            
    def replace(self, d):
        return d.get(self, self)

    def __contains__(self, item):
        return self == item

    @property
    def hasTrial(self):
        return False
    
    @property
    def rhs(self):
        return self

    @property
    def lhs(self):
        return NGSFunction()
        
    @property
    def isNonlinear(self):
        return False
        
    @property
    def isTimeDependent(self):
        return self._tdep

    def __str__(self):
        return self._name


class VolumeField(NGSFunctionBase):
    """
    NGSFunction that provide the access to the element wise volume.
    Args:
        name(str): The symbol name
    """
    def __init__(self, name=None):
        self._name = name
        self._init = False

    def _initialize(self, fes):
        sp = ngsolve.L2(fes.mesh)
        self._func = ngsolve.GridFunction(sp)
        data = ngsolve.Integrate(ngsolve.CoefficientFunction(1), fes.mesh, element_wise=True)
        self._func.vec.data = np.array(data)
        self._init = True

    @property
    def shape(self):
        return ()

    @property
    def valid(self):
        return True
    
    def eval(self, fes):
        if not self._init:
            self._initialize(fes)
        return self._func
            
    def grad(self, fes):
        raise RuntimeError("Gradient of the element-wise volume is not defined.")
            
    def replace(self, d):
        return d.get(self, self)

    def __contains__(self, item):
        return self == item

    @property
    def hasTrial(self):
        return False
    
    @property
    def rhs(self):
        return self

    @property
    def lhs(self):
        return NGSFunction()
        
    @property
    def isNonlinear(self):
        return False
        
    @property
    def isTimeDependent(self):
        return False

    def __str__(self):
        return self._name
    

class SolutionFieldFunction(NGSFunction):
    pass